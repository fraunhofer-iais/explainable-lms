from abc import ABC, abstractmethod
from typing import List


class Perturber(ABC):
    @abstractmethod
    def perturb(self, text: str, features: List[str]) -> List[str]:
        ...
