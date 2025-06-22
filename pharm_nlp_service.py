from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
import logging
import json
import signal
import sys
from typing import Dict, Optional
from embedding_utils import EmbeddingComparator
from pharm_name_service import PharmaNameService

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

logging.basicConfig(
    filename='service.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

DEFAULT_SIMILARITY_THRESHOLD = 0.8

#MODEL_PATH = "./all-MiniLM-L6-v2"
MODEL_PATH_EMB = "./paraphrase-multilingual-MiniLM-L12-v2"
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



#################################################### HTTP methods #################################################################

@app.post("/predict")
async def predict_endpoint(request: ProductRequest):
    result = service.process_name(
        request.product,
        use_cache=request.use_cache,
        similarity_threshold=request.similarity_threshold
    )
    return result

@app.post("/batch_predict")
async def batch_predict_endpoint(request: BatchProductRequest):
    result = service.process_names_batch(
        request.products, 
        use_cache=request.use_cache,
        similarity_threshold=request.similarity_threshold)
    return result    


@app.post("/compare_parts", response_model=ComparePartsResponse)
async def compare_parts(request: ComparePartsRequest):
    similarities = {}
    for field, ref_value in request.reference.items():
        test_value = request.test.get(field, "")

        sim = comparator.compare_fields(field, ref_value, test_value)
        similarities[field] = round(sim, 4)

    return ComparePartsResponse(similarities=similarities)

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
