# embedding_utils.py
from sentence_transformers import SentenceTransformer
import numpy as np
import re



class EmbeddingComparator:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def cosine_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))

    @staticmethod
    def compare_numbers(text1, text2):
        nums1 = set(re.findall(r'\d+(?:\.\d+)?', text1))
        nums2 = set(re.findall(r'\d+(?:\.\d+)?', text2))
        return 1.0 if nums1 == nums2 else 0.0

    @staticmethod
    def compare_units(text1, text2):
        def normalize_units(t):
            t = t.lower()
            t = t.replace('мг', 'мг').replace('мл', 'мл').replace('ед', 'ед')
            return t.strip()
        return 1.0 if normalize_units(text1) == normalize_units(text2) else 0.0

    def select_compare_function(self, field_name: str):
        # Набор правил: ключевые слова и соответствующие функции
        rules = [
            (r'количеств|qty|число|count', self.compare_numbers),
            (r'ед|unit|единиц', self.compare_units),
            # Добавьте другие правила здесь
        ]
        field_name_lower = field_name.lower()

        for pattern, func in rules:
            if re.search(pattern, field_name_lower):
                return func
        # По умолчанию универсальное сравнение
        return None

    def compare_fields(self, field_name: str, ref_value: str, test_value: str):
        func = self.select_compare_function(field_name)
        if func:
            return func(ref_value, test_value)
        else:
            emb_ref, emb_test = self.embed([ref_value, test_value])
            return self.cosine_similarity(emb_ref, emb_test)

    @staticmethod
    def extract_numbers(text):
        pattern = r'\d+(?:\.\d+)?'
        return re.findall(pattern, text)