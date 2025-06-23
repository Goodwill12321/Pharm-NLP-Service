
import logging
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from cache_nlp_results import PharmaNameCache  # модифицированный класс из шага 1
import time
from embedding_utils import EmbeddingComparator


DEFAULT_SIMILARITY_THRESHOLD = 0.8



class PharmaNameService:
    def __init__(self, config, embedding_comparator: EmbeddingComparator):
        
        AUTOSAVE_INTERVAL = config.get("faiss_autosave_interval", 300)  # по умолчанию 5 минут
        self.cache = PharmaNameCache(embedding_comparator, autosave_interval=AUTOSAVE_INTERVAL)
        MODEL_PATH = config.get("model_path", "Goodwill333/T5-NER-Pharm-RU")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используем устройство: {self.device}")

        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH,  cache_dir=".").to(self.device)
        logging.info("Модель и токенизатор загружены")

    def predict(self, product: str) -> str:
        input_text = "\n".join([
            "Задание: Извлеки части из названия лекарственного препарата или товара фармацевтического назначения.",
            f"Наименование товара: {product}"
        ])
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True).to(self.device)
        start_time = time.time()
        outputs = self.model.generate(**inputs, max_length=256, num_beams=2, early_stopping=True)
        end_time = time.time()
        generation_time = end_time - start_time
        message = f"⏱️ Время генерации: {generation_time:.2f} секунд"
        logging.info(message)
        print(message)
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        message2 = f"Наименование товара: {product}\n🔍 decoded: {decoded}"
        logging.info(message2)
        print(message2)
        return decoded.replace("\n", " ").strip()


    def predict_batch(self, products: list[str]) -> list[str]:
        input_texts = [
            "\n".join([
                "Задание: Извлеки части из названия лекарственного препарата или товара фармацевтического назначения.",
                f"Наименование товара перечисленыы ниже (в возвращаемом результате добавляй в json поле ТоварПоставки препарат, по которому было извлечение ): {product}"
            ])
            for product in products
        ]
        inputs = self.tokenizer(
            input_texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=256
        ).to(self.device)
        
        start_time = time.time()
        
        outputs = self.model.generate(
            **inputs, 
            max_length=256, 
            num_beams=2, 
            early_stopping=True
        )
        end_time = time.time()
        generation_time = end_time - start_time
        message = f"⏱️ Время генерации по батчу размером {len(products)}: {generation_time:.2f} секунд"
        logging.info(message)
        print(message)
       
        result = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            .replace("\n", " ")
            .strip()
            for output in outputs
        ]
        return result


    def process_name(self, name, use_cache=True, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
        if use_cache:
            cached_result = self.cache.query_cache(name, similarity_threshold=similarity_threshold)
            if cached_result is not None:
                parsed, cached_name, dist = cached_result
                print(f"Используем кэш для: {name} (найдено: {cached_name}, dist={dist:.3f})")
                return {"parsed": parsed, "cached_name": cached_name, "from_cache": True, "similarity": float(dist)}

        print(f"Обрабатываем моделью T5: {name}")
        result = self.predict(name)
        self.cache.add_to_cache(name, result)
        return {"parsed": result, "cached_name": None, "from_cache": False, "similarity": None}
    
    def process_names_batch(self, products, use_cache=True, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD):
        uncached_indices = []
        uncached_products = []
        results = [None] * len(products)
        for idx, product in enumerate(products):
            if use_cache:
                cached = self.cache.query_cache(product, similarity_threshold=similarity_threshold)
            if use_cache and (cached is not None):
                parsed, cached_name, dist = cached
                results[idx] ={"parsed": parsed, "cached_name": cached_name, "from_cache": True, "similarity": float(dist)}
            else:
                uncached_indices.append(idx)
                uncached_products.append(product)
        # Для всех не найденных в кэше — батчевое предсказание
        if uncached_products:
            batch_results = self.predict_batch(uncached_products)
            for i, product, result in zip(uncached_indices, uncached_products, batch_results):
                self.cache.add_to_cache(product, result)
                #cached_results.append({"parsed": result, "cached_name": None, "from_cache": False, "similarity": None})
                results[i] = {"parsed": result, "cached_name": None, "from_cache": False, "similarity": None}
        return {"results": results}
    

   
