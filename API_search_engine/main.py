from fastapi import FastAPI
from controllers.router import router as search_router

app = FastAPI(
    title="Semantic News Search API",
    version="0.5.0",
    description="FAISS / Qdrant / TF-IDF / BM25 backends + REST"
)

app.include_router(search_router, prefix="/api")




@app.get("/ping")
async def ping():
    return {"status": "ok"}