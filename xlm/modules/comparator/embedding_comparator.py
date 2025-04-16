from typing import List
from scipy import spatial
from xlm.modules.comparator.comparator import Comparator
from xlm.components.encoder.encoder import Encoder
from xlm.utils.scores import normalize_scores, reverse_scores


class EmbeddingComparator(Comparator):
    """
    We use cosine similarity to compare two texts. The base for the embeddings can be any LLM.
    """

    def __init__(self, encoder: Encoder):
        self.__encoder = encoder

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        embeddings = self.__encoder.encode(texts=texts + [reference_text])
        vectors: List[List[float]] = embeddings[:-1]
        ref_vector: List[float] = embeddings[-1]
        scores = []
        for vector in vectors:
            scores.append(self.__get_cosine_similarity(x=vector, y=ref_vector))
        if do_normalize_scores:
            scores = normalize_scores(scores=scores)
        scores = reverse_scores(scores=scores)
        return scores

    def __get_cosine_similarity(self, x: List[float], y: List[float]):
        return 1 - spatial.distance.cosine(x, y)
