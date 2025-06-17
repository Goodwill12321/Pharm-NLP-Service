from fastapi import FastAPI
from pydantic import BaseModel
from cache_nlp_results import PharmaNameCache
import logging
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Логирование в файл
logging.basicConfig(
    filename='service.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

#MODEL_PATH = "./t5-med-ner"  # Укажите путь к вашей модели
MODEL_PATH = "Goodwill333/T5-NER-Pharm-RU"

# Загрузка модели и токенизатора при старте
logging.info("Загрузка модели и токенизатора...")
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH,  cache_dir=".").to(device)
logging.info("Модель и токенизатор загружены")

class ProductRequest(BaseModel):
    product: str

def predict(product: str) -> str:
    input_text = "\n".join([
        "Задание: Извлеки части из названия лекарственного препарата или товара фармацевтического назначения.",
        f"Наименование товара: {product}"
    ])
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
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


@app.post("/predict")
async def predict_endpoint(request: ProductRequest):
    result = service.process_name(request.product)
    return {"result": result}




# Основной сервис с кэшем
class PharmaNameService:
    def __init__(self):
        self.cache = PharmaNameCache()

    def process_name(self, name):
        # Сначала пытаемся получить результат из кэша
        cached_result = self.cache.query_cache(name)
        if cached_result is not None:
            print(f"Используем кэш для: {name}")
            return cached_result
        
        # Если нет подходящего кэша — вызываем модель T5
        print(f"Обрабатываем моделью T5: {name}")
        result = predict(name)
        
        # Добавляем результат в кэш
        self.cache.add_to_cache(name, result)
        return result





import uvicorn

if __name__ == "__main__":
    service = PharmaNameService()
    names = [
        "Парацетамол 500мг",
        "Парацетамол 650мг",
        "Ибупрофен 200мг",
        "Парацетамол 500мг таблетки"
    ]
    
    for n in names:
        parts = service.process_name(n)
        print(parts)

    uvicorn.run("pharm_nlp_service:app", host="127.0.0.1", port=8000, reload=True)
