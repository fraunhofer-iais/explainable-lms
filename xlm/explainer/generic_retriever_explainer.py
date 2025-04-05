from typing import List, Tuple

from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import ExplanationGranularity
from xlm.explainer.generic_explainer import GenericExplainer
from xlm.modules.comparator.comparator import Comparator
from xlm.modules.perturber.perturber import Perturber
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer
from xlm.modules.tokenizer.tokenizer import Tokenizer


class GenericRetrieverExplainer(GenericExplainer):
    def __init__(
        self,
        perturber: Perturber,
        comparator: Comparator,
        retriever: Retriever,
        tokenizer: Tokenizer = CustomTokenizer(),
    ):
        super().__init__(
            tokenizer=tokenizer, perturber=perturber, comparator=comparator
        )
        self.retriever = retriever

    def get_features(
        self,
        input_text: str,
        reference_text: str,
        reference_score: float,
        granularity: ExplanationGranularity,
    ):
        features = self.tokenizer.tokenize(text=reference_text, granularity=granularity)
        return features

    def get_perturbations(
        self,
        input_text: str,
        reference_text: str,
        features: List[str],
    ):
        perturbations = self.perturber.perturb(text=reference_text, features=features)
        return perturbations

    def get_reference(self, input_text: str) -> Tuple[str, float]:
        retrieved_documents, reference_scores = self.retriever.retrieve(
            text=input_text, top_k=3, return_scores=True
        )
        return retrieved_documents[0], reference_scores[0]

    def get_post_perturbation_results(self, input_text: str, perturbations: List[str]):
        perturbed_document_embeddings = self.retriever.encoder.encode(
            texts=perturbations
        )
        self.retriever.corpus_embeddings = perturbed_document_embeddings
        self.retriever.corpus_documents = perturbations

        retrieved_perturbed_documents, scores_with_perturbed_documents = (
            self.retriever.retrieve(
                text=input_text,
                top_k=len(self.retriever.corpus_documents),
                return_scores=True,
            )
        )

        return scores_with_perturbed_documents

    def get_comparator_scores(
        self,
        reference_text: str,
        reference_score: float,
        results: List[str] | List[float],
        do_normalize_scores: bool,
    ):
        scores = self.comparator.compare(
            reference_text=reference_score,
            texts=results,
            do_normalize_scores=do_normalize_scores,
        )
        return scores
