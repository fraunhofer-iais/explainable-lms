from typing import List, Union
import textdistance as td
from xlm.modules.comparator.comparator import Comparator
from xlm.utils.scores import normalize_scores, reverse_scores


class GenericComparator(Comparator):
    """
    Class to use any compare functions. Inspiration taken from https://yassineelkhal.medium.com/the-complete-guide-to-string-similarity-algorithms-1290ad07c6b7#:~:text=N%2Dgram%20similarity,extracted%20from%20a%20given%20string.

    So far we can pick:
    Hamming: overlapping chars
    Levenshtein: insert, delete, substitute
    Damerau_Levenshtein: levenshtein + swap adjacent chars
    Jaro: damerau_levenshtein, but the chars don't have to be adjacent
    Jaro_Winkler: jaro with more reward for strings starting with the same chars
    Smith_Waterman: find the optimal local alignment between two sequences
    Lcsseq: longest common subsequence
    Lcsstr: longest common substring
    """

    def __init__(
        self,
        similarity_fn: Union[
            td.hamming.normalized_similarity,
            td.levenshtein.normalized_similarity,
            td.damerau_levenshtein.normalized_similarity,
            td.jaro.normalized_similarity,
            td.jaro_winkler.normalized_similarity,
            td.smith_waterman.normalized_similarity,
            td.lcsseq.normalized_similarity,
            td.lcsstr.normalized_similarity,
        ],
    ):
        self.__similarity_fn = similarity_fn

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        scores = [self.__similarity_fn(reference_text, text) for text in texts]
        if do_normalize_scores:
            scores = normalize_scores(scores=scores)
        scores = reverse_scores(scores)
        return scores


class LevenshteinComparator(GenericComparator):
    def __init__(self):
        super().__init__(td.levenshtein.normalized_similarity)


class JaroWinklerComparator(GenericComparator):
    def __init__(self):
        super().__init__(td.jaro_winkler.normalized_similarity)
