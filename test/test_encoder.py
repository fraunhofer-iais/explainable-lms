from unittest.mock import MagicMock
import pytest
from requests import Session
import numpy as np
from xlm.components.encoder.encoder import Encoder
from test.utils import random_vector


@pytest.fixture()
def session():
    session = MagicMock(spec=Session)
    session.post = MagicMock()
    session.post.return_value.status_code = 200
    dimension = 384
    session.post.return_value.json = MagicMock()
    session.post.return_value.json.return_value = [
        random_vector(seed=42, dimension=dimension),
        random_vector(seed=24, dimension=dimension),
    ]
    return session


@pytest.fixture()
def encoder(session):
    return Encoder(model_name="sentence-transformers", endpoint="", session=session)


@pytest.fixture()
def texts():
    return ["Hello World.", "Have you ever seen explanations for an LLM?"]


def test_encoder(encoder, texts):
    encoded_text = encoder.encode(texts)
    assert isinstance(encoded_text, list)
    assert isinstance(encoded_text[0], list)
    assert isinstance(encoded_text[1], list)
    assert isinstance(encoded_text[0][0], np.float64)
    assert isinstance(encoded_text[1][0], np.float64)
    assert len(encoded_text) == len(texts)
    assert len(encoded_text[0]) == len(encoded_text[1]) == 384
