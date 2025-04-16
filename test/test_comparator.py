from unittest.mock import MagicMock
from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
from xlm.modules.comparator.generic_comparator import (
    JaroWinklerComparator,
    LevenshteinComparator,
)
from xlm.modules.comparator.n_gram_overlap_comparator import NGramOverlapComparator
from xlm.components.encoder.encoder import Encoder
import pytest
import numpy as np
from test.utils import random_vector


@pytest.fixture()
def jaro_winkler_comparator():
    return JaroWinklerComparator()


@pytest.fixture()
def levenshtein_comparator():
    return LevenshteinComparator()


@pytest.fixture()
def reference_text():
    return "I am a big fan"


@pytest.fixture()
def texts():
    return ["I am a tennis fan"]


@pytest.fixture()
def encoder():
    encoder = MagicMock(spec=Encoder)
    encoder.encode = MagicMock()
    dimension = 384
    emb_a = random_vector(seed=42, dimension=dimension)
    emb_b = random_vector(seed=24, dimension=dimension)
    encoder.encode.return_value = [emb_a, emb_b]
    return encoder


@pytest.fixture()
def embedding_comparator(encoder):
    return EmbeddingComparator(encoder)


@pytest.fixture()
def extraction_score_comparator():
    return ExtractionScoreComparator()


@pytest.fixture()
def n_gram_overlap_comparator():
    return NGramOverlapComparator()


def test_jaro_winkler_comparator(jaro_winkler_comparator, reference_text, texts):
    result = jaro_winkler_comparator.compare(reference_text, texts)
    assert isinstance(result, list)
    assert result[0] >= 0
    assert np.round(result[0], 4) == 0.1207


def test_levenshtein_comparator(levenshtein_comparator, reference_text, texts):
    result = levenshtein_comparator.compare(reference_text, texts)
    assert isinstance(result, list)
    assert result[0] >= 0
    assert np.round(result[0], 4) == 0.2941


def test_embedding_comparator(embedding_comparator, reference_text, texts):
    result = embedding_comparator.compare(reference_text, texts)
    assert isinstance(result, list)
    assert result[0] >= 0
    assert np.round(result[0], 4) == 1.0471


def test_n_gram_overlap_comparator(n_gram_overlap_comparator, reference_text, texts):
    result = n_gram_overlap_comparator.compare(reference_text, texts)
    assert isinstance(result, list)
    assert result[0] >= 0
    assert np.round(result[0], 4) == 0.6667
