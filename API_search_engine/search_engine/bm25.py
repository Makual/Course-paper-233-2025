# search_engine/bm25.py
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from .base import SearchBackend
from typing import List, Tuple

class BM25Search(SearchBackend):
    def __init__(self):
        self.tokenized = []
        self.doc_ids = []
        self.bm25 = None

    def index(self, texts: List[str], ids: List[str]):
        self.doc_ids = ids
        self.tokenized = [word_tokenize(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_tok = word_tokenize(query.lower())
        scores = self.bm25.get_scores(q_tok)
        top_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_ids]
