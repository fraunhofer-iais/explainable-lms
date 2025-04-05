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
        granularity: ExplanationGranularity,
        reference: str | Tuple[str, float],
    ):
        features = self.tokenizer.tokenize(text=reference[0], granularity=granularity)
        return features

    def get_perturbations(
        self,
        input_text: str,
        reference: str | Tuple[str, float],
        features: List[str],
    ):
        perturbations = self.perturber.perturb(text=reference[0], features=features)
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
        reference: str | float,
        results: List[str] | List[float],
        do_normalize_scores: bool,
    ):
        scores = self.comparator.compare(
            reference_text=reference[1],
            texts=results,
            do_normalize_scores=do_normalize_scores,
        )
        return scores

    # def explain(
    #     self,
    #     user_input: str,
    #     granularity: ExplanationGranularity,
    #     model_name: Optional[str] = None,
    #     do_normalize_comparator_scores: bool = True,
    #     system_response: Optional[str] = None,
    # ) -> ExplanationDto | List[ExplanationDto]:
    #     input_text = user_input
    #
    #     retrieved_documents, reference_scores = self.retriever.retrieve(
    #         text=input_text, top_k=3, return_scores=True
    #     )
    #
    #     explanation_dtos = []
    #     for retrieved_document, reference_score in zip(
    #         retrieved_documents, reference_scores
    #     ):
    #         document_features = self.tokenizer.tokenize(
    #             text=retrieved_document, granularity=granularity
    #         )
    #
    #         perturbations = self.perturber.perturb(
    #             text=retrieved_document, features=document_features
    #         )
    #
    #         perturbed_document_embeddings = self.retriever.encoder.encode(
    #             texts=perturbations
    #         )
    #         self.retriever.corpus_embeddings = perturbed_document_embeddings
    #         self.retriever.corpus_documents = perturbations
    #
    #         retrieved_perturbed_documents, scores_with_perturbed_documents = (
    #             self.retriever.retrieve(
    #                 text=input_text,
    #                 top_k=len(self.retriever.corpus_documents),
    #                 return_scores=True,
    #             )
    #         )
    #
    #         scores = self.comparator.compare(
    #             reference_text=reference_score,
    #             texts=scores_with_perturbed_documents,
    #             do_normalize_scores=do_normalize_comparator_scores,
    #         )
    #
    #         explanation_dto = self.__get_explanation_dto(
    #             features=document_features,
    #             scores=scores,
    #             input_text=input_text,
    #             output_text=retrieved_document,
    #         )
    #
    #         explanation_dtos.append(explanation_dto)
    #
    #     return explanation_dtos
