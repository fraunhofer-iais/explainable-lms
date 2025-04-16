from typing import List

from xlm.modules.comparator.comparator import Comparator
import numpy as np

from xlm.utils.scores import normalize_scores, reverse_scores


class ScoreComparator(Comparator):
    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        delta_scores = np.array(reference_text) - np.array(texts)
        if do_normalize_scores:
            delta_scores = normalize_scores(scores=delta_scores)

        delta_scores = reverse_scores(scores=delta_scores)
        return delta_scores
