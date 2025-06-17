import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class PharmaNameCache:
    def __init__(self, embedding_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', dim=384):
        # Загружаем модель эмбеддингов
        self.embedder = SentenceTransformer(embedding_model_name)
        self.dim = dim
        
        # Инициализация Faiss индекса для поиска по косинусной близости
        # Для косинусной близости нормируем векторы и используем Inner Product
        self.index = faiss.IndexFlatIP(dim)
        
        # Словарь для хранения оригинальных наименований и результатов разбиения
        self.data = []
        self.embeddings = []

    def add_to_cache(self, name, parsed_parts):
        emb = self.embedder.encode([name], normalize_embeddings=True)[0]
        self.index.add(np.array([emb], dtype='float32'))
        self.data.append({'name': name, 'parsed': parsed_parts})
        self.embeddings.append(emb)

    def query_cache(self, name, top_k=3, similarity_threshold=0.7):
        emb = self.embedder.encode([name], normalize_embeddings=True)[0].astype('float32')
        if self.index.ntotal == 0:
            return None  # Кэш пуст
        
        D, I = self.index.search(np.array([emb]), top_k)  # Поиск top_k ближайших соседей
        for dist, idx in zip(D[0], I[0]):
            if dist >= similarity_threshold:
                cached_entry = self.data[idx]
                # Здесь можно добавить дополнительный анализ различий между name и cached_entry['name']
                # и попытаться адаптировать parsed_parts по аналогии
                return cached_entry['parsed']
        return None


export = PharmaNameCache