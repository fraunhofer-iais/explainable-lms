import datetime
from unittest.mock import MagicMock
import pytest
from requests import Session
from xlm.components.generator.llm_generator import LLMGenerator


@pytest.fixture
def generator():
    return LLMGenerator(
        session=Session(),
        endpoint="http://localhost:9985",
        num_threads=10,
        model_name="gpt-2",
    )


@pytest.fixture
def generator_wo_multithread():
    return LLMGenerator(
        session=Session(),
        endpoint="http://localhost:9985",
        num_threads=1,
        model_name="gpt-2",
    )


@pytest.fixture()
def mock_session():
    session = MagicMock(spec=Session)
    session.post = MagicMock()
    session.post.return_value.status_code = 200
    session.post.return_value.json = MagicMock()
    session.post.return_value.json.return_value = ["Nope, never heard..."]
    return session


@pytest.fixture()
def llm_generator(mock_session):
    return LLMGenerator(
        session=mock_session, endpoint="", model_name="luminous-supreme-control"
    )


@pytest.fixture()
def texts():
    return ["Have you ever heard something about explainable language models?"]


@pytest.mark.e2etest
def test_generator(generator):
    start = datetime.datetime.now()
    texts = ["hi", "my name", "how are"]
    response = generator.generate(texts=texts)
    end = datetime.datetime.now()
    print("Time taken:", end - start)
    assert len(response) == 3


@pytest.mark.e2etest
def test_generator_wo_multithread(generator_wo_multithread):
    start = datetime.datetime.now()
    texts = ["hi", "my name", "how are"]
    response = generator_wo_multithread.generate(texts=texts)
    end = datetime.datetime.now()
    print("Time taken:", end - start)
    assert len(response) == 3


def test_llm_generator(llm_generator, texts):
    result = llm_generator.generate(texts=texts)
    assert isinstance(result, list)
    assert len(result) == len(texts)
