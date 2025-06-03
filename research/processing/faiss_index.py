import os
import faiss
import numpy as np
from typing import List, Tuple
from huggingface_hub import InferenceClient

from processing.utils import clean_text


HF_API_KEY = os.getenv("HF_TOKEN")
client = InferenceClient(provider="auto", api_key=HF_API_KEY)

class FaissResearch:
    def __init__(self, model_name: str = "sergeyzh/BERTA", embed_dim: int = 768):
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.index_faiss = None
        self.doc_ids = []
        self.embeddings = None

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        cleaned = [clean_text(t) for t in texts]
        resp = client.feature_extraction(cleaned, model=self.model_name)
        arr = np.array(resp, dtype=np.float32)

        # L2 normalisation
        norms = np.linalg.norm(arr, axis=1, keepdims=True)

        arr = arr / (norms + 1e-12)
        return arr

    def index(self, texts: List[str], ids: List[str]):
        vectors = self._get_embeddings(texts)
        self.doc_ids = ids
        self.embeddings = vectors

        dim = vectors.shape[1]

        self.index_faiss = faiss.IndexFlatL2(dim)

        self.index_faiss.add(self.embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self._get_embeddings([query])

        D, I = self.index_faiss.search(q_vec, top_k)
        results: List[Tuple[str, float]] = []
        for rank, idx in enumerate(I[0]):
            score = float(D[0][rank])
            doc_id = self.doc_ids[idx]
            results.append((doc_id, score))
        return results
