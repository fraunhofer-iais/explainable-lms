from xlm.components.encoder.encoder import Encoder
from xlm.components.generator.llm_generator import LLMGenerator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.sbert_retriever import SBERTRetriever

if __name__ == "__main__":
    encoder_model_name = "sentence-transformers"
    generator_model_name = "mistral-7b"
    lms_endpoint = "http://localhost:9985"

    with open("data/climate_change.txt", encoding="utf-8") as f:
        data = f.readlines()

    corpus_documents = [item for item in data if item.strip()]

    # user_input = "What is the primary driver of the current warming trend in the Earth's climate system, according to the essay?"
    user_input = "How does the essay describe the impact of climate change on precipitation patterns?"

    prompt_template = "Context: {context}\nQuestion: {question}\nAnswer:"

    encoder = Encoder(model_name=encoder_model_name, endpoint=lms_endpoint)
    retriever = SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)

    generator = LLMGenerator(endpoint=lms_endpoint, model_name=generator_model_name)

    system = RagSystem(
        retriever=retriever,
        generator=generator,
        prompt_template=prompt_template,
        retriever_top_k=1,
    )

    rag_output = system.run(user_input=user_input)
    print(rag_output.model_dump_json(indent=4))
