from abc import ABC, abstractmethod
from typing import List, Tuple

class SearchBackend(ABC):
    @abstractmethod
    def index(self, texts: List[str], ids: List[str]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        pass