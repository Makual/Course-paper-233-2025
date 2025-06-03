from __future__ import annotations

import os
from typing import List

from sentence_transformers import SentenceTransformer

# ➊ Попытка импортировать клиент Voyage
try:
    import voyageai  # pip install voyageai>=0.1.7
except ImportError as exc:   # пусть остальная часть проекта отвалится заметно
    voyageai = None

# ➋ Карта «локальная-модель» → «HF-ID или placeholder»
MODELS: dict[str, str] = {
    "berta": "sergeyzh/BERTA",
    # для Voyage оставляем «пустышку»; сам репозиторий нам теперь не нужен
    "voyage": "<remote>",
}

# ➌ Обёртка, поведенчески совместимая с SentenceTransformer
class VoyageModel:  # pylint: disable=too-few-public-methods
    """
    Мини-адаптер, чтобы .encode(...) работал так же, как у Sentence-Transformers.
    Под капотом дергает Voyage API.
    """

    def __init__(
        self,
        model_name: str = "voyage-3.5",          # выбери свою модель
        input_type: str = "document",
        api_key: str | None = None,
        batch_size: int = 96,                    # можно тюнить под rate-limit
    ) -> None:
        if voyageai is None:
            raise ImportError(
                "voyageai не установлен; pip install voyageai"
            ) from None

        self.client = voyageai.Client(api_key or os.getenv("VOYAGE_API_KEY"))
        self.model_name = model_name
        self.input_type = input_type
        self.batch_size = batch_size

    # ключевое: интерфейс .encode, который ждёт весь остальной код
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:  # noqa: D401
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            resp = self.client.embed(
                chunk,
                model=self.model_name,
                input_type=self.input_type,
                **kwargs,                       # passthrough (truncation, dtype…)
            )
            embeddings.extend(resp.embeddings)
        return embeddings

    # полезно, если где-то используется эта информация
    @property
    def embedding_dimension(self) -> int:
        # один запрос в API, чтобы узнать размерность
        meta = self.client.models.retrieve(self.model_name)
        return meta.output_dimension


# ➍ Универсальный лоадер
def load_model(name: str):
    """
    Возвращает либо SentenceTransformer, либо VoyageModel,
    но с одинаковым методом .encode(...)
    """
    if name == "voyage":
        return VoyageModel()           # берёт ключ из VOYAGE_API_KEY
    if name not in MODELS:
        raise ValueError(f"Неизвестная модель: {name}")
    return SentenceTransformer(MODELS[name])