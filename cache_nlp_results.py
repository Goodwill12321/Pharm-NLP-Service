import faiss
import numpy as np
import threading
import time
import json
import os
from embedding_utils import EmbeddingComparator

class PharmaNameCache:
    def __init__(self, embedding_comparator: EmbeddingComparator,
                 dim=384,
                 index_path="faiss.index", data_path="cache_data.json", autosave_interval=300):
        self.embedding_comparator = embedding_comparator
        self.dim = dim
        self.index_path = index_path
        self.data_path = data_path
        self.autosave_interval = autosave_interval

        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            print("Загрузка индекса и кэша с диска...")
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            #self.embeddings = [np.array(v, dtype='float32') for v in [entry.get("embedding", None) for entry in self.data]]
        else:
            print("Создание нового индекса и кэша...")
            self.index = faiss.IndexFlatIP(dim)
            self.data = []
            #self.embeddings = []

        self._stop_event = threading.Event()
        self._autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self._autosave_thread.start()

    def add_to_cache(self, name, parsed_parts):
        emb = self.embedding_comparator.embed([name])[0].astype('float32')
        self.index.add(np.array([emb]))
        self.data.append({'name': name, 'parsed': parsed_parts})
        #self.embeddings.append(emb)

    def query_cache(self, name, top_k=3, similarity_threshold=0.8):
        emb = self.embedding_comparator.embed([name])[0].astype('float32')
        if self.index.ntotal == 0:
            return None

        D, I = self.index.search(np.array([emb]), top_k)
        query_numbers = self.embedding_comparator.extract_numbers(name)

        candidates = []
        for dist, idx in zip(D[0], I[0]):
            if dist >= similarity_threshold:
                cached_entry = self.data[idx]
                cached_numbers = self.embedding_comparator.extract_numbers(cached_entry['name'])
                if query_numbers == cached_numbers:
                    candidates.append((dist, cached_entry))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_dist, best_entry = candidates[0]
        return best_entry['parsed'], best_entry['name'], best_dist

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Кэш и индекс сохранены на диск")

    def _autosave_loop(self):
        while not self._stop_event.is_set():
            time.sleep(self.autosave_interval)
            try:
                self.save()
            except Exception as e:
                print(f"Ошибка при автосохранении кэша: {e}")

    def stop_autosave(self):
        self._stop_event.set()
        self._autosave_thread.join()
        self.save()
