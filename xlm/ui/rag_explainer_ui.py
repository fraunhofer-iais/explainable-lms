from typing import List, Optional

import gradio as gr
from gradio.components import Markdown

from xlm.components.rag_system.rag_system import RagSystem
from xlm.dto.dto import ExplanationDto, ExplanationGranularity
from xlm.explainer.generic_generator_explainer import GenericGeneratorExplainer
from xlm.explainer.generic_retriever_explainer import GenericRetrieverExplainer
from xlm.registry.comparators import load_comparator
from xlm.registry.perturbers import load_perturber
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
        rag_system: RagSystem,
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

        self.rag_system = rag_system

        self.retriever_explainer = self.get_retriever_explainer()
        self.generator_explainer = self.get_generator_explainer()

        self.app: gr.Blocks = self.build_app()

    def build_app(self):
        with gr.Blocks(
            theme=gr.themes.Monochrome(
                font=[gr.themes.GoogleFont("Quicksand"), "Arial", "sans-serif"]
            ).set(
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
                # granularity,
                # upper_percentile,
                # middle_percentile,
                # lower_percentile,
                # explainer_name,
                # model_name,
                # perturber_name,
                # comparator_name,
                submit_btn,
                user_input_text,
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
                    # granularity,
                    # upper_percentile,
                    # middle_percentile,
                    # lower_percentile,
                    # explainer_name,
                    # model_name,
                    # perturber_name,
                    # comparator_name,
                ],
                outputs=[
                    user_input_text,
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
        if len(user_input) == 0:
            gr.Error("Please provide an input!")
            return None

        rag_output = self.rag_system.run(user_input=user_input)

        retriever_explanation_granularity = ExplanationGranularity.WORD_LEVEL
        retriever_explanation_dto = self.retriever_explainer.explain(
            user_input=user_input,
            reference_text=rag_output.retrieved_documents[0],
            reference_score=rag_output.retriever_scores[0],
            granularity=retriever_explanation_granularity,
            do_normalize_comparator_scores=True,
        )

        retriever_explanations_vis = self.__visualize_explanations(
            text_to_visualize=rag_output.retrieved_documents[0],
            explanation_dto=retriever_explanation_dto,
            granularity=retriever_explanation_granularity,
            upper_percentile=85,
            middle_percentile=75,
            lower_percentile=10,
        )

        generator_explanation_granularity = ExplanationGranularity.SENTENCE_LEVEL
        generator_explanation_dto = self.generator_explainer.explain(
            user_input=rag_output.prompt,
            reference_text=rag_output.generated_responses[0],
            reference_score=None,
            granularity=generator_explanation_granularity,
            do_normalize_comparator_scores=True,
        )

        generator_explanations_vis = self.__visualize_explanations(
            text_to_visualize=rag_output.prompt,
            explanation_dto=generator_explanation_dto,
            granularity=generator_explanation_granularity,
            upper_percentile=85,
            middle_percentile=65,
            lower_percentile=5,
        )

        retrieved_document = rag_output.retrieved_documents[0]
        prompt = rag_output.prompt
        generated_response = rag_output.generated_responses[0]

        return (
            user_input,
            retrieved_document,
            prompt,
            generated_response,
            retriever_explanations_vis,
            generator_explanations_vis,
        )

    def get_retriever_explainer(self) -> GenericRetrieverExplainer:
        retriever_perturber = load_perturber(
            perturber_name=self.retriever_perturber_name
        )
        retriever_comparator = load_comparator(
            comparator_name=self.retriever_comparator_name
        )

        retriever_explainer = GenericRetrieverExplainer(
            perturber=retriever_perturber,
            comparator=retriever_comparator,
            retriever=self.rag_system.retriever,
        )

        return retriever_explainer

    def get_generator_explainer(self) -> GenericGeneratorExplainer:
        generator_perturber = load_perturber(
            perturber_name=self.generator_perturber_name
        )
        generator_comparator = load_comparator(
            comparator_name=self.generator_comparator_name
        )

        generator_explainer = GenericGeneratorExplainer(
            perturber=generator_perturber,
            comparator=generator_comparator,
            generator=self.rag_system.generator,
        )

        return generator_explainer

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
            user_input = gr.Textbox(
                placeholder="Type your question here and press Enter.",
                label="Question",
                container=True,
                lines=1,
            )
        # with gr.Row():
        #     granularity = gr.Radio(
        #         choices=[e for e in ExplanationGranularity],
        #         value=ExplanationGranularity.SENTENCE_LEVEL,
        #         label="Explanation Granularity",
        #     )

        # with gr.Accordion(label="Settings", open=False, elem_id="accordion"):
        # with gr.Row(variant="compact"):
        #     explainer_name = gr.Radio(
        #         label="Explainer",
        #         choices=list(EXPLAINERS.keys()),
        #         value=list(EXPLAINERS.keys())[0],
        #         container=True,
        #     )
        # with gr.Row(variant="compact"):
        #     upper_percentile = gr.Textbox(label="Upper", value="85", container=True)
        #     middle_percentile = gr.Textbox(
        #         label="Middle", value="75", container=True
        #     )
        #     lower_percentile = gr.Textbox(label="Lower", value="10", container=True)

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
                with gr.Column(scale=1):
                    gr.Markdown("🔍 **User Query** (input to the Retriever)")
                    with gr.Row():
                        user_input_text = gr.Markdown(container=True)

                    gr.Markdown("📝 **Retrieved document** (output from the Retriever)")
                    with gr.Row():
                        retrieved_document = gr.Markdown(container=True)

                with gr.Column(scale=1):
                    gr.Markdown("✨ **Explaining the Retrieval process**")
                    with gr.Row():
                        retriever_vis = gr.HTML(container=True)

        with gr.Accordion(
            label="Generate and Explain!", open=False, elem_id="accordion"
        ):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("❓ **Prompt** (input to the Generator)")
                    with gr.Row():
                        prompt = gr.Markdown(container=True)

                    gr.Markdown("✔️ **Generated Response** (output from the Generator)")
                    with gr.Row():
                        generated_response = gr.Markdown(container=True)

                with gr.Column(scale=1):
                    gr.Markdown("✨ **Explaining the Generation process**")
                    with gr.Row():
                        generator_vis = gr.HTML(container=True)

        return (
            user_input,
            # granularity,
            # upper_percentile,
            # middle_percentile,
            # lower_percentile,
            # explainer_name,
            # model_name,
            # perturber_name,
            # comparator_name,
            submit_btn,
            user_input_text,
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
        granularity: ExplanationGranularity,
        upper_percentile: Optional[int] = 85,
        middle_percentile: Optional[int] = 75,
        lower_percentile: Optional[int] = 10,
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
            avoid_exp_label=True,
            avoid_legend=True,
            granularity=granularity,
        )
