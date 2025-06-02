# search_engine/tfidf.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base import SearchBackend
from typing import List, Tuple

class TfidfSearch(SearchBackend):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=50000)
        self.texts = []
        self.doc_ids = []
        self.matrix = None

    def index(self, texts: List[str], ids: List[str]):
        self.texts = texts
        self.doc_ids = ids
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_ids = scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_ids]
