import os
from typing import List, Tuple
from huggingface_hub import InferenceClient

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams, SearchParams

from processing.utils import clean_text

HF_API_KEY = os.getenv("HF_TOKEN")
client = InferenceClient(provider="auto", api_key=HF_API_KEY)

class QdrantResearch:
    def __init__(self, collection_name: str = "news_research", model_name: str = "sergeyzh/BERTA", embed_dim: int = 768):
        self.collection = collection_name
        self.model_name = model_name

        self.client_q = QdrantClient(host="localhost", port=6333)
        self.client_q.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        cleaned = [clean_text(t) for t in texts]
        vecs = client.feature_extraction(cleaned, model=self.model_name)

        import numpy as np
        arr = np.array(vecs, dtype=np.float32)
        arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return arr.tolist()

    def index(self, texts: List[str], ids: List[str]):
        vecs = self._get_embeddings(texts)
        points = [
            PointStruct(id=ids[i], vector=vecs[i])
            for i in range(len(ids))
        ]

        self.client_q.upsert(
            collection_name=self.collection,
            points=points,
            wait=True
        )

    def search(self, query: str, top_k: int = 10):
        vec = client.feature_extraction([clean_text(query)], model=self.model_name)[0]
        res = self.client_q.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128),
        )

        return [(r.id.replace("-", ""), r.score) for r in res]