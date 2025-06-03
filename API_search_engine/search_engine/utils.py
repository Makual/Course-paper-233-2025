# researches/utils.py
import os
import re
import nltk
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
STOP_RU = set(nltk.corpus.stopwords.words("russian"))

load_dotenv()
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB   = os.getenv("PG_DB")

DATABASE_URL = (
    f"postgresql://{PG_USER}:{PG_PASS}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
)
engine = create_engine(DATABASE_URL)


def fetch_last_news(limit: int = 10_000) -> pd.DataFrame:
    sql = f"""
        SELECT id,
               title,
               anons,
               body,
               date_creation
          FROM public.tmp_news
         LIMIT {limit};
    """


    dfs = []
    for chunk in pd.read_sql(sql, engine, chunksize=1000):
        dfs.append(chunk)

    if not dfs:
        return pd.DataFrame(columns=["id", "title", "anons", "body", "date_creation"])

    df = pd.concat(dfs, ignore_index=True)
    return df


def fetch_random_news(limit: int = 10_000, sample_pct: float = 1.0) -> pd.DataFrame:
    pct = max(min(sample_pct, 100.0), 0.1)

    sql = f"""
        SELECT id, title, anons, body
          FROM public.tmp_news
         TABLESAMPLE SYSTEM ({pct})
         LIMIT {limit};
    """

    dfs = []
    for chunk in pd.read_sql(sql, engine, chunksize=1000):
        dfs.append(chunk)

    if not dfs:
        return pd.DataFrame(columns=["id", "title", "anons", "body"])

    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_text(txt: str, keep_digits: bool = False) -> str:
    txt = txt.lower()
    if keep_digits:
        txt = re.sub(r"[^а-яё0-9\s]+", " ", txt)
    else:
        txt = re.sub(r"[^а-яё\s]+", " ", txt)
    tokens = [w for w in txt.split() if w not in STOP_RU and len(w) > 2]
    return " ".join(tokens)
