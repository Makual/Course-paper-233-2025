from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from typing import List, Tuple
from utils import clean_text


class BM25Research():
    def __init__(self):
        self.bm25, self.doc_ids, self.tokens = None, [], []

    def index(self, texts: List[str], ids: List[str]):
        self.tokens = [word_tokenize(clean_text(t)) for t in texts]
        self.bm25 = BM25Okapi(self.tokens)
        self.doc_ids = ids

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        q_tok = word_tokenize(clean_text(query))
        scores = self.bm25.get_scores(q_tok)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in idx]