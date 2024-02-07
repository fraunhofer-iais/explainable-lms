from typing import List, Optional, Tuple

from comparator.comparator import Comparator
from dto.dto import (
    ExplanationDto,
    ExplanationGranularity,
    FeatureImportance,
)
from explainer.explainer import Explainer
from generator.generator import Generator
from perturber.perturber import Perturber
from tokenizer.tokenizer import Tokenizer
from utils.scores import sort_similarity_scores


class GenericExplainer(Explainer):
    def __init__(
        self,
        tokenizer: Tokenizer,
        perturber: Perturber,
        generator: Generator,
        comparator: Comparator,
        num_threads: Optional[int] = 10,
    ):
        self.__tokenizer = tokenizer
        self.__perturber = perturber
        self.__generator = generator
        self.__comparator = comparator
        self.__num_threads = num_threads

    def explain(
        self,
        user_input: str,
        granularity: ExplanationGranularity,
        model_name: Optional[str] = None,
        do_normalize_comparator_scores: bool = True,
        system_response: Optional[str] = None,
    ) -> ExplanationDto:
        input_text = user_input

        reference_response = (
            self.__generator.generate(texts=[input_text], model_name=model_name)[0]
            if not system_response
            else system_response
        )

        input_features = self.__tokenizer.tokenize(
            text=input_text, granularity=granularity
        )

        perturbations = self.__perturber.perturb(
            text=input_text, features=input_features
        )

        responses = self.__generator.generate(
            texts=perturbations, model_name=model_name
        )

        scores = self.__comparator.compare(
            reference_text=reference_response,
            texts=responses,
            do_normalize_scores=do_normalize_comparator_scores,
        )

        explanation_dto = self.__get_explanation_dto(
            features=input_features,
            scores=scores,
            input_text=input_text,
            output_text=reference_response,
        )

        return explanation_dto

    def __get_explanation_dto(
        self,
        features: List[str],
        scores: List[float],
        input_text: str,
        output_text: str,
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
