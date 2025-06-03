from __future__ import annotations
import json, os, pickle, sys, tempfile
from pathlib import Path
from typing import List
import faiss
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tqdm import tqdm
from nltk.tokenize import word_tokenize


LIMIT = 10000

BASE = Path(__file__).resolve().parents[1]        
os.chdir(BASE)                                  
BM25_FILE   = BASE / "API_search_engine/data/bm25_state.pkl"
FAISS_FILE  = BASE / "API_search_engine/data/faiss.index"
FAISS_IDS   = BASE / "API_search_engine/data/faiss_doc_ids.json"

if not (BM25_FILE.exists() and FAISS_FILE.exists() and FAISS_IDS.exists()):
    sys.exit("Old files not found")

load_dotenv()
PG_DSN = (
    f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASS')}"
    f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
)
engine = create_engine(PG_DSN, pool_pre_ping=True)

sys.path.append(str(BASE / "API_search_engine"))
from controllers.router import DOC_STORE, bm25_backend, faiss_backend, _embed

with BM25_FILE.open("rb") as f:
    data = pickle.load(f)
bm25_backend.tokens  = data["tokens"] 
bm25_backend.doc_ids = data["doc_ids"]

faiss_backend.index_faiss = faiss.read_index(str(FAISS_FILE))
faiss_backend.doc_ids     = json.loads(FAISS_IDS.read_text())
faiss_backend.embeddings  = np.vstack(
    [faiss_backend.index_faiss.reconstruct(i)
     for i in range(faiss_backend.index_faiss.ntotal)]
).astype("float32")

known_ids: set[str] = set(bm25_backend.doc_ids)
print(f"Loaded index with {len(known_ids):,} docs")


def clean(txt: str) -> str:
    import re, string
    return re.sub(r"\s+", " ",
           re.sub(rf"[{re.escape(string.punctuation)}]", " ", txt.lower())
           ).strip()

def chunked(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf; buf=[]
    if buf: yield buf


SQL = text("""
SELECT id, title, anons, body, date_creation
  FROM public.tmp_news
 WHERE id <> ALL(:ids)                  --   not in индекс
 ORDER BY random()
 LIMIT :lim
""")

BATCH_DB=500; BATCH_EMB=64; BATCH_INS=2000
added_rows: List[dict] = []
added_cnt  = 0

with engine.begin() as connR:
    rows = connR.execution_options(stream_results=True)\
                .execute(SQL, {"ids": list(known_ids), "lim": LIMIT})

    for sample in tqdm(chunked(rows, BATCH_DB), desc="Append"):
        txts, meta = [], []
        for r in sample:
            txts.append(f"{r.title}. {r.anons or ''}. {r.body}")
            meta.append((r.id, r.date_creation, r.title, r.anons, r.body))

        vecs = np.vstack([_embed(txts[i:i+BATCH_EMB])
                          for i in range(0, len(txts), BATCH_EMB)])

        faiss_backend.index_faiss.add(vecs.astype("float32"))
        faiss_backend.embeddings = np.vstack([faiss_backend.embeddings, vecs])
        faiss_backend.doc_ids.extend([m[0] for m in meta])

        for (doc_id, dt, title, anons, body), full, vec in zip(meta, txts, vecs):
            bm25_backend.tokens.append(word_tokenize(clean(full)) or ["dummy"]) 
            bm25_backend.doc_ids.append(doc_id)                                 
            DOC_STORE[doc_id] = {"clean": full, "date": dt, "vec": vec}
            added_rows.append({"id": doc_id, "title": title,
                               "anons": anons, "body": body, "date": dt})

        if len(added_rows) >= BATCH_INS:
            with engine.begin() as c:
                c.execute(text(
                    "INSERT INTO public.news_indexed "
                    "(id,title,anons,body,date_creation) "
                    "VALUES (:id,:title,:anons,:body,:date)"
                ), added_rows)
            added_cnt += len(added_rows); added_rows.clear()


if added_rows:
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO public.news_indexed "
            "(id,title,anons,body,date_creation) "
            "VALUES (:id,:title,:anons,:body,:date)"
        ), added_rows)
    added_cnt += len(added_rows)


from rank_bm25 import BM25Okapi
bm25_backend.bm25 = BM25Okapi(bm25_backend.tokens)  

def atomic_write(path: Path, data: bytes | str):
    tmp = Path(tempfile.mkstemp(dir=path.parent, prefix=".tmp_")[1])
    with tmp.open("wb" if isinstance(data, bytes) else "w", encoding=None if isinstance(data, bytes) else "utf-8") as f:
        f.write(data)
    tmp.replace(path)

atomic_write(BM25_FILE, pickle.dumps(
    {"tokens": bm25_backend.tokens, "doc_ids": bm25_backend.doc_ids})
)
atomic_write(FAISS_IDS, json.dumps(faiss_backend.doc_ids, ensure_ascii=False))
faiss.write_index(faiss_backend.index_faiss, str(FAISS_FILE))

print(f"Added {added_cnt:,} new docs")