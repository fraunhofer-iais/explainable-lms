from typing import List, Tuple
import numpy as np


def normalize_scores(scores: List[float]) -> List[float]:
    if (np.max(scores) - np.min(scores)) == 0:
        return scores
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))


def sort_similarity_scores(
        features: List[str], scores: List[float]
) -> Tuple[List[str], List[float]]:
    scores_sort_idxes = np.asarray(scores).argsort()
    scores_sorted = [scores[idx] for idx in scores_sort_idxes]
    features_sorted = [features[idx] for idx in scores_sort_idxes]

    return features_sorted, scores_sorted


def reverse_scores(scores: List[float]) -> List[float]:
    return [1 - sim for sim in scores]
