from abc import ABC, abstractmethod
from typing import List


class Comparator(ABC):
    @abstractmethod
    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        """
        Returns
        -------
        List[float] Scores between 0 and 1. The higher the score, the more the sentences differ.
        """
        ...
