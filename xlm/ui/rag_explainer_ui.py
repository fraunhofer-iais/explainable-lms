from typing import List, Optional

import gradio as gr
from gradio.components import Markdown

from xlm.components.encoder.encoder import Encoder
from xlm.components.generator.llm_generator import LLMGenerator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.dto.dto import ExplanationDto, ExplanationGranularity
from xlm.explainer.generic_generator_explainer import GenericGeneratorExplainer
from xlm.explainer.generic_retriever_explainer import GenericRetrieverExplainer
from xlm.modules.registry.comparators import load_comparator
from xlm.modules.registry.perturbers import load_perturber
from xlm.utils.categorizer import PercentileBasedCategorizer
from xlm.utils.visualizer import Visualizer


class RagExplainerUI:
    def __init__(
        self,
        logo_path: str,
        css_path: str,
        visualizer: Visualizer,
        window_title: str,
        title: str,
        examples: Optional[List[str]] = None,
    ):
        self.__logo_path = logo_path
        self.__css_path = css_path
        self.__examples = examples
        self.__window_title = window_title
        self.__title = title
        self.__visualizer = visualizer

        self.retriever_perturber_name = "leave_one_out"
        self.retriever_comparator_name = "score_comparator"

        self.generator_perturber_name = "leave_one_out"
        self.generator_comparator_name = "sentence_transformers_based_comparator"

        self.encoder_model_name = "sentence-transformers"
        self.generator_model_name = "mistral-7b"
        self.lms_endpoint = "http://localhost:9985"
        self.data_path = "data/climate_change.txt"

        self.retriever = self.get_retriever()
        self.retriever_explainer = self.get_retriever_explainer()

        self.generator = self.get_generator()
        self.generator_explainer = self.get_generator_explainer()

        self.rag_system = self.get_rag_system()

        self.app: gr.Blocks = self.build_app()

    def build_app(self):
        with gr.Blocks(
            theme=gr.themes.Monochrome().set(
                button_primary_background_fill="#009374",
                button_primary_background_fill_hover="#009374C4",
                checkbox_label_background_fill_selected="#028A6EFF",
            ),
            css=self.__css_path,
            title=self.__window_title,
        ) as demo:
            self.__build_app_title()
            (
                user_input,
                granularity,
                upper_percentile,
                middle_percentile,
                lower_percentile,
                # explainer_name,
                # model_name,
                # perturber_name,
                # comparator_name,
                submit_btn,
                retrieved_document,
                prompt,
                generated_response,
                retriever_vis,
                generator_vis,
            ) = self.__build_chat_and_explain()

            submit_btn.click(
                fn=self.run,
                inputs=[
                    user_input,
                    granularity,
                    upper_percentile,
                    middle_percentile,
                    lower_percentile,
                    # explainer_name,
                    # model_name,
                    # perturber_name,
                    # comparator_name,
                ],
                outputs=[
                    retrieved_document,
                    prompt,
                    generated_response,
                    retriever_vis,
                    generator_vis,
                ],
            )

        return demo

    def run(
        self,
        user_input: str,
        granularity: ExplanationGranularity,
        upper_percentile: str,
        middle_percentile: str,
        lower_percentile: str,
        perturber_name: str,
        comparator_name: str,
    ):
        rag_output = self.rag_system.run(user_input=user_input)

        retriever_explanation_dto = self.retriever_explainer.explain(
            user_input=user_input,
            reference_text=rag_output.retrieved_documents[0],
            reference_score=rag_output.retriever_scores[0],
            granularity=granularity,
            do_normalize_comparator_scores=True,
        )

        generator_explanation_dto = self.generator_explainer.explain(
            user_input=user_input,
            reference_text=rag_output.generated_responses[0],
            reference_score=None,
            granularity=granularity,
            do_normalize_comparator_scores=True,
        )

        retriever_explanations_vis = self.__visualize_explanations(
            text_to_visualize=rag_output.retrieved_documents[0],
            explanation_dto=retriever_explanation_dto,
            upper_percentile=int(upper_percentile),
            middle_percentile=int(middle_percentile),
            lower_percentile=int(lower_percentile),
        )

        generator_explanations_vis = self.__visualize_explanations(
            text_to_visualize=rag_output.prompt,
            explanation_dto=generator_explanation_dto,
            upper_percentile=int(upper_percentile),
            middle_percentile=int(middle_percentile),
            lower_percentile=int(lower_percentile),
        )

        retrieved_document = rag_output.retrieved_documents[0]
        prompt = rag_output.prompt
        generated_response = rag_output.generated_responses[0]

        return (
            retrieved_document,
            prompt,
            generated_response,
            retriever_explanations_vis,
            generator_explanations_vis,
        )

    def get_retriever(self):
        encoder = self.get_encoder()
        with open(self.data_path, encoding="utf-8") as f:
            data = f.readlines()
        corpus_documents = [item.strip() for item in data if item.strip()]
        return SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)

    def get_encoder(self):
        return Encoder(model_name=self.encoder_model_name, endpoint=self.lms_endpoint)

    def get_retriever_explainer(self):
        retriever_perturber = load_perturber(
            perturber_name=self.retriever_perturber_name
        )
        retriever_comparator = load_comparator(
            comparator_name=self.retriever_comparator_name
        )

        retriever_explainer = GenericRetrieverExplainer(
            perturber=retriever_perturber,
            comparator=retriever_comparator,
            retriever=self.retriever,
        )

        return retriever_explainer

    def get_generator(self):
        return LLMGenerator(
            endpoint=self.lms_endpoint, model_name=self.generator_model_name
        )

    def get_generator_explainer(self):
        generator_perturber = load_perturber(
            perturber_name=self.generator_perturber_name
        )
        generator_comparator = load_comparator(
            comparator_name=self.generator_comparator_name
        )

        generator_explainer = GenericGeneratorExplainer(
            perturber=generator_perturber,
            comparator=generator_comparator,
            generator=self.generator,
        )

        return generator_explainer

    def get_rag_system(self):
        prompt_template = "Context: {context}\nQuestion: {question}\n\nAnswer:"
        system = RagSystem(
            retriever=self.retriever,
            generator=self.generator,
            prompt_template=prompt_template,
            retriever_top_k=1,
        )
        return system

    def __build_app_title(self):
        with gr.Row():
            with gr.Column(min_width=50, scale=1):
                gr.Image(
                    value=self.__logo_path,
                    width=50,
                    height=50,
                    show_download_button=False,
                    container=False,
                )
            with gr.Column(scale=2):
                Markdown(
                    f'<p style="text-align: left; font-size:200%; font-weight: bold"'
                    f">{self.__title}"
                    f"</p>"
                )

    def __build_chat_and_explain(self):
        with gr.Row():
            with gr.Column(scale=2):
                user_input = gr.Textbox(
                    placeholder="Type your question here and press Enter.",
                    label="Question",
                    container=True,
                    lines=10,
                )
            with gr.Column(scale=1):
                granularity = gr.Radio(
                    choices=[e for e in ExplanationGranularity],
                    value=ExplanationGranularity.SENTENCE_LEVEL,
                    label="Explanation Granularity",
                )

        with gr.Accordion(label="Settings", open=False, elem_id="accordion"):
            # with gr.Row(variant="compact"):
            #     explainer_name = gr.Radio(
            #         label="Explainer",
            #         choices=list(EXPLAINERS.keys()),
            #         value=list(EXPLAINERS.keys())[0],
            #         container=True,
            #     )
            with gr.Row(variant="compact"):
                upper_percentile = gr.Textbox(label="Upper", value="85", container=True)
                middle_percentile = gr.Textbox(
                    label="Middle", value="75", container=True
                )
                lower_percentile = gr.Textbox(label="Lower", value="10", container=True)

            # with gr.Row(variant="compact"):
            #     model_name = gr.Radio(
            #         label="Model",
            #         choices=list(MODELS.keys()),
            #         value=list(MODELS.keys())[0],
            #         container=True,
            #     )
            # with gr.Row(variant="compact"):
            #     perturber_name = gr.Radio(
            #         label="Perturber",
            #         choices=list(PERTURBERS.keys()),
            #         value=list(PERTURBERS.keys())[0],
            #         container=True,
            #     )
            # with gr.Row(variant="compact"):
            #     comparator_name = gr.Radio(
            #         label="Comparator",
            #         choices=list(COMPARATORS.keys()),
            #         value=list(COMPARATORS.keys())[0],
            #         container=True,
            #     )
        with gr.Row(variant="compact"):
            # passing "elem_id" to use a custom style for the component
            # in the CSS passed.
            submit_btn = gr.Button(
                value="🛠 Submit",
                variant="secondary",
                elem_id="button",
                interactive=True,
            )

        with gr.Accordion(
            label="Retrieve and Explain!", open=False, elem_id="accordion"
        ):
            with gr.Row():
                retrieved_document = gr.Markdown(
                    label="Retrieved Document",
                    container=True,
                    # interactive=False,
                )
            with gr.Row():
                retriever_vis = gr.HTML(label="Retriever Explanations")

        with gr.Accordion(
            label="Generate and Explain!", open=False, elem_id="accordion"
        ):
            with gr.Row():
                prompt = gr.Markdown(
                    label="Prompt to the LLM",
                    container=True,
                    # interactive=False,
                )

            with gr.Row():
                generated_response = gr.Markdown(
                    label="Generated Response",
                    container=True,
                    # interactive=False,
                )

            with gr.Row():
                generator_vis = gr.HTML(label="Generator Explanations")

        return (
            user_input,
            granularity,
            upper_percentile,
            middle_percentile,
            lower_percentile,
            # explainer_name,
            # model_name,
            # perturber_name,
            # comparator_name,
            submit_btn,
            retrieved_document,
            prompt,
            generated_response,
            retriever_vis,
            generator_vis,
        )

    def __visualize_explanations(
        self,
        text_to_visualize: str,
        explanation_dto: ExplanationDto,
        upper_percentile: Optional[int],
        middle_percentile: Optional[int],
        lower_percentile: Optional[int],
    ) -> str:
        segregator = PercentileBasedCategorizer(
            upper_bound_percentile=upper_percentile,
            middle_bound_percentile=middle_percentile,
            lower_bound_percentile=lower_percentile,
        )
        return self.__visualizer.visualize(
            segregator=segregator,
            explanations=explanation_dto,
            output_from_explanations=text_to_visualize,
        )
