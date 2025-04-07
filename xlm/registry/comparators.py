from requests import Session
from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
from xlm.modules.comparator.generic_comparator import (
    LevenshteinComparator,
    JaroWinklerComparator,
)
from xlm.modules.comparator.n_gram_overlap_comparator import NGramOverlapComparator
from xlm.modules.comparator.score_comaprator import ScoreComparator
from xlm.components.encoder.encoder import Encoder
from xlm.registry import DEFAULT_LMS_ENDPOINT
from xlm.registry.encoder import load_encoder

levenshtein_comparator = LevenshteinComparator()
jaro_winkler_comparator = JaroWinklerComparator()
n_gram_comparator = NGramOverlapComparator()
sentence_transformers_based_comparator = EmbeddingComparator(
    encoder=Encoder(
        model_name="sentence-transformers",
        endpoint=DEFAULT_LMS_ENDPOINT,
        session=Session(),
    )
)
score_comparator = ScoreComparator()

COMPARATORS = {
    "sentence_transformers_based_comparator": sentence_transformers_based_comparator,
    "base_llm_based_comparator": ...,  # loaded with each request depending on llm used for generation
    "levenshtein_comparator": levenshtein_comparator,
    "jaro_winkler_comparator": jaro_winkler_comparator,
    "n_gram_comparator": n_gram_comparator,
    "score_comparator": score_comparator,
}


def load_comparator(comparator_name: str, model_name: str = None):
    if comparator_name == "base_llm_based_comparator":
        return EmbeddingComparator(encoder=load_encoder(model_name=model_name))
    else:
        if comparator_name not in COMPARATORS.keys():
            raise Exception(
                f"The entered perturber name is not found! Available "
                f"perturbers are: {list(COMPARATORS.keys())}"
            )

        return COMPARATORS.get(comparator_name)
