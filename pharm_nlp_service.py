from fastapi import FastAPI, Query
from pydantic import BaseModel
import logging
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
import signal
import sys

from cache_nlp_results import PharmaNameCache  # модифицированный класс из шага 1

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

MODEL_PATH = config.get("model_path", "Goodwill333/T5-NER-Pharm-RU")
AUTOSAVE_INTERVAL = config.get("faiss_autosave_interval", 300)  # по умолчанию 5 минут
DEFAULT_SIMILARITY_THRESHOLD = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH,  cache_dir=".").to(device)
logging.info("Модель и токенизатор загружены")


def predict(product: str) -> str:
    input_text = "\n".join([
        "Задание: Извлеки части из названия лекарственного препарата или товара фармацевтического назначения.",
        f"Наименование товара: {product}"
    ])
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    start_time = time.time()
    outputs = model.generate(**inputs, max_length=256, num_beams=2, early_stopping=True)
    end_time = time.time()
    generation_time = end_time - start_time
    message = f"⏱️ Время генерации: {generation_time:.2f} секунд"
    logging.info(message)
    print(message)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    message2 = f"Наименование товара: {product}\n🔍 decoded: {decoded}"
    logging.info(message2)
    print(message2)
    return decoded.replace("\n", " ").strip()

class PharmaNameService:
    def __init__(self):
        #MODEL_PATH = "./all-MiniLM-L6-v2"
        MODEL_PATH = "./paraphrase-multilingual-MiniLM-L12-v2"
        self.cache = PharmaNameCache(embedding_model_name=MODEL_PATH, autosave_interval=AUTOSAVE_INTERVAL)

    def process_name(self, name, use_cache=True, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
        if use_cache:
            cached_result = self.cache.query_cache(name, similarity_threshold=similarity_threshold)
            if cached_result is not None:
                parsed, cached_name, dist = cached_result
                print(f"Используем кэш для: {name} (найдено: {cached_name}, dist={dist:.3f})")
                return {"parsed": parsed, "cached_name": cached_name, "from_cache": True, "similarity": float(dist)}

        print(f"Обрабатываем моделью T5: {name}")
        result = predict(name)
        self.cache.add_to_cache(name, result)
        return {"parsed": result, "cached_name": None, "from_cache": False, "similarity": None}

service = PharmaNameService()

from pydantic import BaseModel, Field

class ProductRequest(BaseModel):
    product: str
    use_cache: bool = Field(True, description="Использовать кэш")
    similarity_threshold: float = Field(DEFAULT_SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Порог похожести для кэша")

@app.post("/predict")
async def predict_endpoint(request: ProductRequest):
    result = service.process_name(
        request.product,
        use_cache=request.use_cache,
        similarity_threshold=request.similarity_threshold
    )
    return result

def shutdown_handler(signum, frame):
    print("Завершение работы сервиса, сохраняем кэш...")
    service.cache.stop_autosave()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

import uvicorn

if __name__ == "__main__":
    service = PharmaNameService()
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
