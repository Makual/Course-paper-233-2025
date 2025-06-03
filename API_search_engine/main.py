from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from controllers.router import router as search_router
from fastapi.responses import FileResponse

app = FastAPI(
    title="Semantic News Search API",
    version="0.5.0",
    description="FAISS + BM25 + augmented retrival search. Based on REST"
)

app.include_router(search_router)
app.mount("/static", StaticFiles(directory="static"), name="static")
 
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")
     
@app.get("/ping")
async def ping():
    return {"status": "ok"}