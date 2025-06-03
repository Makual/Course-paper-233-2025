from __future__ import annotations

import os, json, pickle, enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import faiss, requests
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, text, bindparam,
)
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

BM25_PKL   = "data/bm25_state.pkl"
FAISS_IDX  = "data/faiss.index"
FAISS_EMB  = "data/faiss_emb.npy"
FAISS_IDS  = "data/faiss_doc_ids.json"
TOP_BM25   = 100 
MODEL_NAME = "sergeyzh/BERTA"
EMB_DIM    = 768
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"


load_dotenv()
PG_DSN = (
    f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASS')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)
engine = create_engine(PG_DSN, future=True)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.news_indexed (
            id  TEXT PRIMARY KEY,
            title TEXT,
            anons TEXT,
            body  TEXT,
            date_creation TIMESTAMP
        )
    """))


SQL_COUNT = text("SELECT COUNT(*) FROM public.news_indexed")
SQL_LIST  = text("""
    SELECT id, title, date_creation
      FROM public.news_indexed
  ORDER BY date_creation DESC
     LIMIT :lim OFFSET :off
""")
SQL_ONE   = text("SELECT * FROM public.news_indexed WHERE id=:id")
def sql_by_ids(n):
    return (
        text(f"""
            SELECT id, title, date_creation
              FROM public.news_indexed
             WHERE id IN :ids
        """).bindparams(bindparam("ids", expanding=True))
    )




from search_engine.utils import clean_text
from search_engine.bm25 import BM25Search
from search_engine.faiss_index import FaissSearch

bm25_backend = BM25Search()
faiss_backend = FaissSearch(model_name=MODEL_NAME, embed_dim=EMB_DIM)

DOC_STORE: Dict[str, Dict] = {}  


def _load_bm25():
    p = Path(BM25_PKL)
    if not p.exists():
        return
    data = pickle.loads(p.read_bytes())
    bm25_backend.tokens   = data["tokens"]
    bm25_backend.doc_ids  = data["doc_ids"]
    bm25_backend.bm25     = BM25Okapi(bm25_backend.tokens)

def _load_faiss():
    idx_f, ids_f = Path(FAISS_IDX), Path(FAISS_IDS)

    if not (idx_f.exists() and ids_f.exists()):
        return

    faiss_backend.index_faiss = faiss.read_index(str(idx_f))
    faiss_backend.doc_ids     = json.loads(ids_f.read_text())

    embs = [faiss_backend.index_faiss.reconstruct(i)
            for i in range(faiss_backend.index_faiss.ntotal)]
    faiss_backend.embeddings = np.vstack(embs).astype("float32")

def _hydrate_doc_dates():
    if DOC_STORE:
        return
    with engine.connect() as conn:
        for r in conn.execute(text("SELECT id, date_creation FROM public.news_indexed")).mappings():
            DOC_STORE[r["id"]] = {"date": r["date_creation"]}

_load_bm25()
_load_faiss()
_hydrate_doc_dates()

def _embed(texts: List[str]) -> np.ndarray:
    return faiss_backend._get_embeddings(texts)




OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

def gpt_expand(q: str, k=3) -> str:
    if not OPENROUTER_KEY:
        return ""
    sys_prompt = (
        "Ты ассистент для расширения и исправления запроса пользователя к новостному порталу vesti.ru, чтобы улучшить поиск"
        "Добавь альтернативные варианты к запросу, поправив орфографию если надо, дополнив смысл, добавляя деталей нужных"
        "Если запрос явно с опечаткой, поменяй на наиболее вероятную"
        f"Дай {k} альтернатив через запятую. Без лишних комментариев"
    )
    payload = {
        "model":"openai/gpt-4o-mini",
        "messages":[
            {"role":"system","content":"Всегда отвечай на русском. "+sys_prompt},
            {"role":"user","content":q}
        ],
        "temperature":0,"max_tokens":128,"seed":42,
    }
    headers = {"Authorization":f"Bearer {OPENROUTER_KEY}","Content-Type":"application/json"}
    try:
        r = requests.post(OPENROUTER_API, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        print(r.json()["choices"][0]["message"]["content"])
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("GPT augmentaion fail:", e)
        return ""


class NewsInput(BaseModel):
    id: str
    title: str
    anons: Optional[str]
    body: str
    date_creation: datetime = Field(default_factory=datetime.utcnow)

class SearchResponseItem(BaseModel):
    id: str
    score: float

class NewsShort(BaseModel):
    id: str
    title: str
    date_creation: datetime

class NewsFull(BaseModel):
    id: str
    title: str
    anons: Optional[str]
    body: str
    date_creation: datetime

class NewsPage(BaseModel):
    page: int
    page_size: int
    total: int
    items: List[NewsShort]

router = APIRouter(prefix="/api", tags=["search"])


def hybrid_search(q: str, aug_q: str, *, top_k: int = 10) -> List[Tuple[str, float]]:
    bm  = bm25_backend.bm25.get_scores(word_tokenize(clean_text(q)))
    idx = np.argsort(bm)[::-1][:TOP_BM25]
    ids_main = [bm25_backend.doc_ids[i] for i in idx]

    bm_a  = bm25_backend.bm25.get_scores(word_tokenize(clean_text(aug_q)))
    idx_a = np.argsort(bm_a)[::-1][:TOP_BM25]
    ids_aug = [bm25_backend.doc_ids[i] for i in idx_a]


    cand_ids = list(dict.fromkeys(ids_main + ids_aug))


    if faiss_backend.index_faiss is None:
        return [(doc_id,
                 float(bm[bm25_backend.doc_ids.index(doc_id)]))
                for doc_id in cand_ids[:top_k]]

    common_ids = [i for i in cand_ids if i in faiss_backend.doc_ids]
    if not common_ids:                   # FAISS не знает ни одного из кандидатов
        return [(doc_id,
                 float(bm[bm25_backend.doc_ids.index(doc_id)]))
                for doc_id in cand_ids[:top_k]]


    q_vec = _embed([q]).astype("float32")[0]
    sub_idxs = [faiss_backend.doc_ids.index(i) for i in common_ids]
    sub_embs = faiss_backend.embeddings[sub_idxs]
    scores   = sub_embs @ q_vec
    order    = scores.argsort()[::-1][:top_k]

    return [(common_ids[i], float(scores[i])) for i in order]


@router.get("/search", response_model=List[SearchResponseItem], summary="Поиск")
async def search(
    q: str,
    from_date: Optional[datetime] = Query(None),
    to_date:   Optional[datetime] = Query(None),
    top_k: int = 10,
):
    if not DOC_STORE:
        raise HTTPException(503, "index empty")

    aug_q = gpt_expand(q).strip()
    results = hybrid_search(q, aug_q, top_k=top_k)

    if from_date or to_date:
        results = [
            (doc_id, score) for doc_id, score in results
            if (from_date is None or DOC_STORE[doc_id]["date"] >= from_date) and
               (to_date   is None or DOC_STORE[doc_id]["date"] <= to_date)
        ]

    if not results:
        raise HTTPException(404, "No results")

    return [SearchResponseItem(id=d, score=s) for d, s in results]


def _persist_news(news: NewsInput, vec: np.ndarray):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO public.news_indexed (id,title,anons,body,date_creation)
            VALUES (:id,:title,:anons,:body,:dt)
            ON CONFLICT (id) DO NOTHING
        """), dict(id=news.id, title=news.title, anons=news.anons,
                   body=news.body, dt=news.date_creation))
    DOC_STORE[news.id] = {"date": news.date_creation, "vec": vec}

@router.post("/news", response_model=dict)
async def add_news(news: NewsInput):
    text_raw = f"{news.title}. {news.anons or ''}. {news.body}"
    vec      = _embed([text_raw])[0]

    if faiss_backend.index_faiss is None:
        faiss_backend.index([text_raw], [news.id])
    else:
        faiss_backend.index_faiss.add(vec.reshape(1,-1).astype("float32"))
        faiss_backend.embeddings = np.vstack([faiss_backend.embeddings, vec])
        faiss_backend.doc_ids.append(news.id)

    faiss.write_index(faiss_backend.index_faiss, FAISS_IDX)
    Path(FAISS_IDS).write_text(json.dumps(faiss_backend.doc_ids))

    toks = word_tokenize(clean_text(text_raw)) or ["dummy"]
    bm25_backend.tokens.append(toks)
    bm25_backend.doc_ids.append(news.id)
    bm25_backend.bm25 = BM25Okapi(bm25_backend.tokens)

    _persist_news(news, vec)
    pickle.dump({"tokens":bm25_backend.tokens,"doc_ids":bm25_backend.doc_ids}, open(BM25_PKL,"wb"))
    return {"status":"indexed","id":news.id}

@router.get("/news", response_model=NewsPage)
async def list_news(page:int=1, page_size:int=50):
    if not (1<=page_size<=200):
        raise HTTPException(400,"bad page_size")
    off = (page-1)*page_size
    with engine.connect() as conn:
        total = conn.execute(SQL_COUNT).scalar_one()
        rows  = conn.execute(SQL_LIST, {"lim":page_size,"off":off}).mappings().all()

    return NewsPage(
        page=page, page_size=page_size, total=total,
        items=[NewsShort(**r) for r in rows]
    )

@router.get("/news/{news_id}", response_model=NewsFull)
async def get_news(news_id: str):
    with engine.connect() as conn:          
        row = conn.execute(SQL_ONE, {"id": news_id}).mappings().first()

    if row is None:
        raise HTTPException(status_code=404, detail="Not found")

    return NewsFull(**row)

@router.post("/news/by_ids", response_model=List[NewsShort])
async def news_by_ids(ids: List[str]):
    stmt = (
        text("""
            SELECT id, title, date_creation
              FROM public.news_indexed
             WHERE id IN :ids
        """).bindparams(bindparam("ids", expanding=True))
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt, {"ids": ids}).mappings().all()

    return [NewsShort(**r) for r in rows]