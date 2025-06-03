# search_engine/models.py
from sentence_transformers import SentenceTransformer

MODELS = {
    "frida": "ai-forever/frida",
    "berta": "sergeyzh/BERTA",
    "frida-mini": "sergeyzh/rubert-mini-frida",
    "voyage": "voyageai/voyage-3-large"
}

def load_model(name: str) -> SentenceTransformer:
    if name == "voyage":
        raise NotImplementedError("Use voyage API instead")
    return SentenceTransformer(MODELS[name])
