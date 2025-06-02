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

def fetch_random_news(limit: int = 10_000, sample_pct: float = 1.0) -> pd.DataFrame:
    """
    Берём «случайную» подвыборку статей (до limit строк) из public.tmp_news:
      - sample_pct — процент (в процентах) TABLESAMPLE SYSTEM (e.g., 1.0 означает 1% строк).
      - limit      — максимальное число строк в итоговом DataFrame.

    Реализовано через TABLESAMPLE SYSTEM + LIMIT, и читаем чанками (chunksize), 
    чтобы PostgreSQL не отправлял слишком много данных сразу.
    """
    # Гарантируем, что 0 < sample_pct <= 100
    pct = max(min(sample_pct, 100.0), 0.1)

    sql = f"""
        SELECT id, title, anons, body
          FROM public.tmp_news
         TABLESAMPLE SYSTEM ({pct})
         LIMIT {limit};
    """

    # Считываем чанками по 1000 строк, чтобы не «сломать» соединение
    dfs = []
    for chunk in pd.read_sql(sql, engine, chunksize=1000):
        dfs.append(chunk)

    if not dfs:
        # Если выборка TABLESAMPLE ничего не вернула, возвращаем пустой DataFrame с нужными колонками
        return pd.DataFrame(columns=["id", "title", "anons", "body"])

    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_text(txt: str, keep_digits: bool = False) -> str:
    """
    Простой препроцессинг русского текста:
    - перевод в нижний регистр
    - удаляем всё, кроме русских букв и (опционально) цифр
    - убираем стоп-слова и слова короче 3 символов
    """
    txt = txt.lower()
    if keep_digits:
        txt = re.sub(r"[^а-яё0-9\s]+", " ", txt)
    else:
        txt = re.sub(r"[^а-яё\s]+", " ", txt)
    tokens = [w for w in txt.split() if w not in STOP_RU and len(w) > 2]
    return " ".join(tokens)

def spell_correct(text: str) -> str:
    """
    Заглушка для спелл-коррекции (например, Yandex.Speller или enchant/jamspell).
    Пока возвращаем текст без изменений.
    """
    return text
