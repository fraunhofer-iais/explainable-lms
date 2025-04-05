from abc import ABC, abstractmethod
from typing import List

from xlm.dto.dto import ExplanationGranularity


class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str, granularity: ExplanationGranularity) -> List[str]:
        ...
