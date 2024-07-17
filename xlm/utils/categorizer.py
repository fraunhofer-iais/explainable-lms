from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np
from xlm.dto.dto import ExplanationDto


class Categorizer(ABC):
    @abstractmethod
    def categorize(
            self, explanations: ExplanationDto
    ) -> Tuple[List[str], List[str], List[str]]:
        ...


class PercentileBasedCategorizer(Categorizer):
    def __init__(
            self,
            upper_bound_percentile: int = 85,
            middle_bound_percentile: int = 75,
            lower_bound_percentile: int = 10,
    ):
        self.__upper_bound_percentile = upper_bound_percentile
        self.__middle_bound_percentile = middle_bound_percentile
        self.__lower_bound_percentile = lower_bound_percentile

    def categorize(
            self,
            explanations: ExplanationDto,
    ) -> Tuple[List[str], List[str], List[str]]:
        scores = [explanation.score for explanation in explanations.explanations]
        scores = np.asarray(scores)
        upper_bound = np.percentile(scores, self.__upper_bound_percentile)
        mid_bound = np.percentile(scores, self.__middle_bound_percentile)
        lower_bound = np.percentile(scores, self.__lower_bound_percentile)

        pos_features = [
            explanation.feature
            for explanation in explanations.explanations
            if explanation.score >= upper_bound and explanation.score != 0
        ]
        mid_features = [
            explanation.feature
            for explanation in explanations.explanations
            if upper_bound > explanation.score >= mid_bound > 0 and explanation.score != 0
        ]
        low_features = [
            explanation.feature
            for explanation in explanations.explanations
            if mid_bound > explanation.score >= lower_bound > 0 and explanation.score != 0
        ]

        return pos_features, mid_features, low_features
