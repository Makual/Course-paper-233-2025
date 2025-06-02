from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import clean_text

class TfidfResearch():
    def __init__(self, max_features: int = 50000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.doc_ids = []
        self.matrix = None

    def index(self, texts: List[str], ids: List[str]):
        # Предобрабатываем тексты
        cleaned = [clean_text(t) for t in texts]
        self.matrix = self.vectorizer.fit_transform(cleaned)
        self.doc_ids = ids

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_cln = clean_text(query)
        q_vec = self.vectorizer.transform([q_cln])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        idx = scores.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]