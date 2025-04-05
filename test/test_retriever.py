import pytest

from xlm.components.encoder.encoder import Encoder
from xlm.components.retriever.sbert_retriever import SBERTRetriever


@pytest.fixture
def corpus_documents():
    return ["hi", "hello"]


@pytest.fixture
def texts(corpus_documents):
    return corpus_documents


@pytest.fixture
def encoder():
    return Encoder(model_name="sentence-transformers")


@pytest.fixture
def retriever(encoder, corpus_documents):
    return SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)


@pytest.fixture
def retriever_with_corpus_embeddings(encoder, corpus_documents):
    corpus_embeddings = encoder.encode(texts=corpus_documents)
    return SBERTRetriever(
        encoder=encoder,
        corpus_documents=corpus_documents,
        corpus_embeddings=corpus_embeddings,
    )


@pytest.mark.e2etest
def test_retriever(retriever, texts):
    documents = retriever.retrieve(text=texts[0], top_k=1)
    assert len(documents) == 1
    assert documents[0] == "hi"

    documents = retriever.retrieve(text=texts[0], top_k=2)
    assert len(documents) == 2
    assert documents[0] == "hi"

    documents = retriever.retrieve(text=texts[0], top_k=3)
    assert len(documents) == 2
    assert documents[0] == "hi"

    documents = retriever.retrieve(text=texts[1], top_k=1)
    assert len(documents) == 1
    assert documents[0] == "hello"


@pytest.mark.e2etest
def test_retriever(retriever_with_corpus_embeddings, texts):
    texts = ["hi", "hello"]

    documents = retriever_with_corpus_embeddings.retrieve(text=texts[0], top_k=1)
    assert len(documents) == 1
    assert documents[0] == "hi"

    documents = retriever_with_corpus_embeddings.retrieve(text=texts[0], top_k=2)
    assert len(documents) == 2
    assert documents[0] == "hi"

    documents = retriever_with_corpus_embeddings.retrieve(text=texts[0], top_k=3)
    assert len(documents) == 2
    assert documents[0] == "hi"

    documents = retriever_with_corpus_embeddings.retrieve(text=texts[1], top_k=1)
    assert len(documents) == 1
    assert documents[0] == "hello"
