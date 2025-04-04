from typing import List, Dict

import torch
from sentence_transformers.util import semantic_search

from xlm.encoder.encoder import Encoder
from xlm.retriever.retriever import Retriever


class SBERTRetriever(Retriever):
    def __init__(
        self,
        encoder: Encoder,
        max_context_length: int = 100,
        num_threads: int = 10,
        corpus_embeddings: List[List[float]] = None,
        corpus_documents: List[str] = None,
    ):
        self.__encoder = encoder

        self.__max_context_length = max_context_length
        self.__num_threads = num_threads

        self.__corpus_documents = corpus_documents

        if not corpus_embeddings:
            self.__corpus_embeddings = self.encode_corpus(texts=corpus_documents)
        else:
            self.__corpus_embeddings = corpus_embeddings

    def retrieve_documents_with_scores(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[str]:
        query_embeddings = self.encode_queries(text=text)
        results = self.search(
            query_embeddings=query_embeddings,
            corpus_embeddings=self.__corpus_embeddings,
            top_k=top_k,
        )
        return results

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[str]:
        result = self.retrieve_documents_with_scores(text=text, top_k=top_k)
        idxes = []
        for item in result[0]:
            idxes.append(item["corpus_id"])
        documents = [self.__corpus_documents[idx] for idx in idxes]
        return documents

    def encode_corpus(self, texts: List[str]) -> List[List[float]]:
        corpus_embeddings = self.__encoder.encode(texts=texts)
        return corpus_embeddings

    def encode_queries(self, text: str) -> List[float]:
        query_embeddings = self.__encoder.encode(texts=[text])
        return query_embeddings

    def search(
        self,
        query_embeddings: List[float],
        corpus_embeddings: List[List[float]],
        top_k: int,
    ) -> List[List[Dict[str, float]]]:
        result = semantic_search(
            query_embeddings=torch.Tensor(query_embeddings),
            corpus_embeddings=torch.Tensor(corpus_embeddings),
            top_k=top_k,
        )
        return result
