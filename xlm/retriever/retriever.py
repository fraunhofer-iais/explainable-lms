from abc import ABC, abstractmethod
from typing import List


class Retriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[str]: ...
