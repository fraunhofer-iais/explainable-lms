from typing import List

from xlm.components.generator.generator import Generator
from xlm.dto.dto import ExplanationGranularity
from xlm.explainer.generic_explainer import GenericExplainer
from xlm.modules.comparator.comparator import Comparator
from xlm.modules.perturber.perturber import Perturber
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer
from xlm.modules.tokenizer.tokenizer import Tokenizer


class GenericGeneratorExplainer(GenericExplainer):
    def __init__(
        self,
        perturber: Perturber,
        comparator: Comparator,
        generator: Generator,
        tokenizer: Tokenizer = CustomTokenizer(),
    ):
        super().__init__(
            tokenizer=tokenizer, perturber=perturber, comparator=comparator
        )
        self.generator = generator

    def get_features(
        self,
        input_text: str,
        reference_text: str,
        reference_score: float,
        granularity: ExplanationGranularity,
    ):
        features = self.tokenizer.tokenize(text=input_text, granularity=granularity)
        return features

    def get_perturbations(
        self,
        input_text: str,
        reference_text: str,
        features: List[str],
    ):
        perturbations = self.perturber.perturb(text=input_text, features=features)
        return perturbations

    def get_reference(self, input_text: str):
        reference_response = self.generator.generate(texts=[input_text])[0]
        return reference_response

    def get_post_perturbation_results(
        self,
        perturbations: List[str],
        input_text: str = None,
    ):
        responses = self.generator.generate(texts=perturbations)
        return responses

    def get_comparator_scores(
        self,
        reference_text: str,
        reference_score: float,
        results: List[str] | List[float],
        do_normalize_scores: bool,
    ):
        scores = self.comparator.compare(
            reference_text=reference_text,
            texts=results,
            do_normalize_scores=do_normalize_scores,
        )
        return scores
