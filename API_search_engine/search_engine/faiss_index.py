# search_engine/faiss_index.py
import faiss
import numpy as np
from .base import SearchBackend
from .models import load_model
from typing import List, Tuple

class FaissSearch(SearchBackend):
    def __init__(self, model_name: str = "frida"):
        self.model = load_model(model_name)
        self.doc_ids = []
        self.index = None
        self.embeddings = None

    def index(self, texts: List[str], ids: List[str]):
        self.doc_ids = ids
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, top_k)
        return [(self.doc_ids[i], float(D[0][j])) for j, i in enumerate(I[0])]
