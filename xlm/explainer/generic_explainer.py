from abc import abstractmethod, ABC
from typing import List, Optional, Tuple

from xlm.dto.dto import ExplanationGranularity, ExplanationDto, FeatureImportance
from xlm.explainer.explainer import Explainer
from xlm.modules.comparator.comparator import Comparator
from xlm.modules.perturber.perturber import Perturber
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer
from xlm.modules.tokenizer.tokenizer import Tokenizer
from xlm.utils.scores import sort_similarity_scores


class GenericExplainer(Explainer):
    def __init__(
        self,
        perturber: Perturber,
        comparator: Comparator,
        tokenizer: Tokenizer = CustomTokenizer(),
        num_threads: Optional[int] = 10,
    ):
        self.tokenizer = tokenizer
        self.perturber = perturber
        self.comparator = comparator
        self.num_threads = num_threads

    @abstractmethod
    def get_reference(self, input_text: str): ...

    @abstractmethod
    def get_features(
        self,
        input_text: str,
        reference_text: str,
        reference_score: float,
        granularity: ExplanationGranularity,
    ): ...

    @abstractmethod
    def get_perturbations(
        self, input_text: str, reference_text: str, features: List[str]
    ): ...

    @abstractmethod
    def get_post_perturbation_results(
        self, input_text: str, perturbations: List[str]
    ): ...

    @abstractmethod
    def get_comparator_scores(
        self,
        reference_text: str,
        reference_score: float,
        results: List[str] | List[float],
        do_normalize_scores: bool,
    ): ...

    def explain(
        self,
        user_input: str,
        granularity: ExplanationGranularity,
        model_name: Optional[str] = None,
        do_normalize_comparator_scores: bool = True,
        reference_text: Optional[str] = None,
        reference_score: Optional[str] = None,
    ) -> ExplanationDto:
        features = self.get_features(
            input_text=user_input,
            reference_text=reference_text,
            reference_score=reference_score,
            granularity=granularity,
        )

        perturbations = self.get_perturbations(
            input_text=user_input,
            reference_text=reference_text,
            features=features,
        )

        responses = self.get_post_perturbation_results(
            input_text=user_input,
            perturbations=perturbations,
        )

        scores = self.get_comparator_scores(
            reference_text=reference_text,
            reference_score=reference_score,
            results=responses,
            do_normalize_scores=do_normalize_comparator_scores,
        )

        explanation_dto = self.__get_explanation_dto(
            features=features,
            scores=scores,
            input_text=user_input,
            # output_text=reference,
        )

        return explanation_dto

    def __get_explanation_dto(
        self,
        features: List[str],
        scores: List[float],
        input_text: str,
        output_text: str = None,
    ) -> ExplanationDto:
        features, scores = self.__sort_scores(features, scores)
        return ExplanationDto(
            explanations=[
                FeatureImportance(feature=feature, score=score)
                for feature, score in zip(features, scores)
            ],
            input_text=input_text,
            output_text=output_text,
        )

    def __sort_scores(
        self, features: List[str], scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        return sort_similarity_scores(features, scores)
