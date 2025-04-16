from abc import ABC, abstractmethod
from typing import List


class Generator(ABC):
    @abstractmethod
    def generate(self, texts: List[str]) -> List[str]: ...
