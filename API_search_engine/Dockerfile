FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git wget && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python - <<'PY'
import nltk, os
os.environ["NLTK_DATA"] = "/usr/local/nltk_data"
nltk.download("punkt")
nltk.download("stopwords")
PY

COPY . .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
