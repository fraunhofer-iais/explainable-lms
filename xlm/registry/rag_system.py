from xlm.components.generator.generator import Generator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.retriever import Retriever


def load_rag_system(
    retriever: Retriever,
    generator: Generator,
    prompt_template: str = "Context: {context}\nQuestion: {question}\n\nAnswer:",
):
    system = RagSystem(
        retriever=retriever,
        generator=generator,
        prompt_template=prompt_template,
        retriever_top_k=1,
    )
    return system
