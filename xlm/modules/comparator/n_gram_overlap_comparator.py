from typing import List, Callable
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from xlm.modules.comparator.comparator import Comparator
from xlm.utils.scores import normalize_scores

nltk.download("punkt")


class NGramOverlapComparator(Comparator):
    """
    Taken from https://mariogarcia.github.io/blog/2021/04/nlp_text_similarity.html
    """

    def __init__(self, n: int = 2, tokenizer: Callable = nltk.word_tokenize):
        self.__tokenize_fn = tokenizer
        self.__n = n

    def compare(
        self, reference_text: str, texts: List[str], do_normalize_scores: bool = True
    ) -> List[float]:
        distances = [self.__distance(reference_text, text) for text in texts]
        if do_normalize_scores:
            distances = normalize_scores(distances)
        return distances

    def __distance(self, reference_text: str, text: str):
        both = [reference_text, text]
        both_tokenized = [self.__tokenize(text) for text in both]
        both_unique_n_grams = [
            self.__unique_n_gram(tokens) for tokens in both_tokenized
        ]
        try:
            distance = jaccard_distance(both_unique_n_grams[0], both_unique_n_grams[1])
            return distance
        except ZeroDivisionError as e:
            # Handle the division by zero case
            print(
                f"Zero Division Error when computing the jaccard_distance, i.e. there are no n_grams, i.e. distance is set to 0, i.e. similarity is set to 1.0"
            )
            return 0.0

    def __tokenize(self, text: str) -> list:
        return self.__tokenize_fn(text)

    def __unique_n_gram(self, tokens: list):
        return set(ngrams(tokens, n=self.__n))

    def test_compare(self, reference_text: str, text: str):
        return self.__distance(reference_text, text)


if __name__ == "__main__":
    sentence_1 = "I am a big fan"
    sentence_2 = "I am a tennis fan"
    comparator = NGramOverlapComparator()
    print(comparator.test_compare(sentence_1, sentence_2))
