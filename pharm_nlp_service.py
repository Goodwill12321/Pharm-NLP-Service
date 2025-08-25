from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import logging
import json
import signal
import sys
from typing import Dict, List
from embedding_utils import EmbeddingComparator
from pharm_name_service import PharmaNameService
import os

# Читаем настройки из файла config.json
def load_config(path="config.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}")
        return {}

config = load_config()


LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
LOG_FILE = os.path.join(LOG_DIR, "service.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

DEFAULT_SIMILARITY_THRESHOLD = 0.8

#MODEL_PATH = "./all-MiniLM-L6-v2"
MODEL_PATH_EMB = "paraphrase-multilingual-MiniLM-L12-v2"
comparator = EmbeddingComparator(MODEL_PATH_EMB)


service = PharmaNameService(config, comparator)


class ProductRequest(BaseModel):
    product: str
    use_cache: bool = Field(True, description="Использовать кэш")
    similarity_threshold: float = Field(DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Порог похожести для кэша")


class ComparePartsRequest(BaseModel):
    reference: Dict[str, str]  # эталонные части
    test: Dict[str, str]       # тестируемые части

class ComparePartsResponse(BaseModel):
    similarities: Dict[str, float]  # поле -> коэффициент сходства (0..1)

class BatchProductRequest(BaseModel):
    products: list[str]
    use_cache: bool = Field(True)
    similarity_threshold: float = Field(DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0)




class BatchComparePartsRequest(BaseModel):
    items: List[ComparePartsRequest]


class BatchComparePartsResponse(BaseModel):
    results: List[ComparePartsResponse]



#################################################### HTTP methods #################################################################

@app.post("/predict")
async def predict_endpoint(request: ProductRequest):
    result = service.process_name(
        request.product,
        use_cache=request.use_cache,
        similarity_threshold=request.similarity_threshold
    )
    return result

@app.post("/predict_batch")
async def batch_predict_endpoint(request: BatchProductRequest):
    result = service.process_names_batch(
        request.products, 
        use_cache=request.use_cache,
        similarity_threshold=request.similarity_threshold)
    return result    


""" @app.post("/compare_parts", response_model=ComparePartsResponse)
async def compare_parts(request: ComparePartsRequest):
    similarities = {}
    for field, ref_value in request.reference.items():
        test_value = request.test.get(field, "")

        sim = comparator.compare_fields(field, ref_value, test_value)
        similarities[field] = round(sim, 4)

    return ComparePartsResponse(similarities=similarities) """


@app.post("/compare_parts", response_model=ComparePartsResponse)
async def compare_parts(request: ComparePartsRequest):
    fields_values = [(field, request.reference.get(field, ""), request.test.get(field, "")) 
                     for field in request.reference.keys()]
    similarities = comparator.compare_fields_batch(fields_values)
    similarities = {k: round(v, 4) for k, v in similarities.items()}
    return ComparePartsResponse(similarities=similarities)



@app.post("/compare_parts_batch", response_model=BatchComparePartsResponse)
async def batch_compare_parts(request: BatchComparePartsRequest):
    results = []
    # Обрабатываем каждый набор в пакете
    for item in request.items:
        fields_values = [(field, item.reference.get(field, ""), item.test.get(field, "")) 
                     for field in item.reference.keys()]
        similarities = comparator.compare_fields_batch(fields_values)
        # Преобразуем в список с сохранением порядка (по reference.keys())
        similarities = {k: round(v, 4) for k, v in similarities.items()}
        results.append(ComparePartsResponse(similarities=similarities))
    return BatchComparePartsResponse(results=results)



""" from typing import List

@app.post("/batch_compare_parts", response_model=List[ComparePartsResponse])
async def batch_compare_parts(request: ComparePartsRequest):
    fields_values = [(idx, field, request.reference.get(field, ""), request.test.get(field, "")) 
                     for idx, field in enumerate(request.reference.keys())]
    similarities_list = comparator.compare_fields_batch(fields_values)
    
    # Формируем массив словарей с полем и значением сходства
    similarities = [{"field": field, "similarity": round(sim, 4)} for field, sim in similarities_list]

    return {"similarities": similarities}
 """

#################################################### End of HTTP methods ############################################################


def shutdown_handler(signum, frame):
    print("Завершение работы сервиса, сохраняем кэш...")
    service.cache.stop_autosave()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

import uvicorn

if __name__ == "__main__":
    service = PharmaNameService(config, comparator)
    names = [
        "Эсслиал Форте (капс. 300 мг №30 яч.конт/п/карт. |М|) Озон-Россия",
        "Эсслиал Форте капсулы 300мг х30 ячейки.конт п карт.",
        "Эсслиал Форте капсулы 30мг х20 ячейки.конт п карт.",
        "Парацетамол 500мг таблетки"
    ]
    
    for n in names:
        parts = service.process_name(n)
        print(parts)

    uvicorn.run("pharm_nlp_service:app", host="0.0.0.0", port=8000, reload=True)
