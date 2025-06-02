from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

from search_engine.tfidf import TfidfSearch
from search_engine.bm25 import BM25Search
from search_engine.faiss_index import FaissSearch
from search_engine.qdrant_index import QdrantSearch


router = APIRouter(tags=["search"])


SEARCH_BACKENDS = {
    "tfidf": TfidfSearch(),
    "bm25":  BM25Search(),
    "faiss": FaissSearch(model_name="frida"),
    "qdrant": QdrantSearch(collection_name="news", model_name="frida")
}


class NewsInput(BaseModel):
    id: str
    title: str
    anons: Optional[str] = None
    body: str
    date_creation: datetime = Field(default_factory=datetime.utcnow)

class SearchResponseItem(BaseModel):
    id: str
    score: float

@router.post("/index", summary="Добавить/переиндексировать новость")
async def add_news(news: NewsInput):
    text = f"{news.title}. {news.anons or ''}. {news.body}"
    for backend in SEARCH_BACKENDS.values():
        backend.index([text], [news.id])          # пополняем каждую базу
    return {"status": "indexed", "id": news.id}

@router.get(
    "/search",
    response_model=List[SearchResponseItem],
    summary="Поиск по новостям"
)
async def search(
    q: str = Query(..., description="Текст запроса"),
    backend: Literal["tfidf", "bm25", "faiss", "qdrant"] = "faiss",
    from_date: Optional[datetime] = Query(None, description="Начало диапазона"),
    to_date:   Optional[datetime] = Query(None, description="Конец диапазона"),
    top_k: int = 10
):
    engine = SEARCH_BACKENDS[backend]

    # TODO: отфильтровать по дате (если храните метаданные отдельно)
    # В демо-варианте даты игнорируются

    results = engine.search(q, top_k=top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No results")
    return [SearchResponseItem(id=r[0], score=r[1]) for r in results]