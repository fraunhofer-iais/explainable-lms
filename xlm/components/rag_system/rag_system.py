from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import RagOutput


class RagSystem:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        prompt_template: str,
        retriever_top_k: int,
    ):
        self.retriever = retriever
        self.generator = generator
        self.prompt_template = prompt_template
        self.retriever_top_k = retriever_top_k

    def run(self, user_input: str) -> RagOutput:
        retrieved_documents, retriever_scores = self.retriever.retrieve(
            text=user_input, top_k=self.retriever_top_k, return_scores=True
        )

        prompt = self.prompt_template.format(
            context="\n".join(retrieved_documents), question=user_input
        )

        generated_responses = self.generator.generate(texts=[prompt])

        return RagOutput(
            retrieved_documents=retrieved_documents,
            retriever_scores=retriever_scores,
            prompt=prompt,
            generated_responses=generated_responses,
            metadata=dict(
                retriever_model_name=self.retriever.encoder.model_name,
                top_k=self.retriever_top_k,
                generator_model_name=self.generator.model_name,
                prompt_template=self.prompt_template,
            ),
        )
