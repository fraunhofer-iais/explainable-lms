from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.registry.encoder import load_encoder


def load_retriever(encoder_model_name: str, lms_endpoint: str, data_path: str):
    encoder = load_encoder(model_name=encoder_model_name, endpoint=lms_endpoint)
    with open(data_path, encoding="utf-8") as f:
        data = f.readlines()
    corpus_documents = [item.strip() for item in data if item.strip()]
    return SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)
