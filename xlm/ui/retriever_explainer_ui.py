from typing import List, Optional
import gradio as gr
from gradio.components import Markdown
from datasets import load_dataset

from xlm.components.encoder.encoder import Encoder
from xlm.explainer.generic_retriever_explainer import GenericRetrieverExplainer
from xlm.modules.registry.comparators import COMPARATORS, load_comparator
from xlm.modules.registry.explainers import EXPLAINERS
from xlm.modules.registry.models import MODELS
from xlm.modules.registry.perturbers import PERTURBERS, load_perturber
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer
from xlm.utils.categorizer import PercentileBasedCategorizer
from xlm.utils.visualizer import Visualizer
from xlm.dto.dto import ExplanationDto, ExplanationGranularity


class RetrieverExplainerUI:
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
                system_response,
                granularity,
                upper_percentile,
                middle_percentile,
                lower_percentile,
                explainer_name,
                model_name,
                perturber_name,
                comparator_name,
                generator_vis,
                submit_btn,
            ) = self.__build_chat_and_explain()

            submit_btn.click(
                fn=self.run,
                inputs=[
                    user_input,
                    granularity,
                    upper_percentile,
                    middle_percentile,
                    lower_percentile,
                    explainer_name,
                    model_name,
                    perturber_name,
                    comparator_name,
                ],
                outputs=[system_response, generator_vis],
            )

        return demo

    def run(
        self,
        user_input: str,
        granularity: ExplanationGranularity,
        upper_percentile: str,
        middle_percentile: str,
        lower_percentile: str,
        explainer_name: str,
        model_name: str,
        perturber_name: str,
        comparator_name: str,
    ):
        tokenizer = CustomTokenizer()

        perturber_name = "leave_one_out"
        comparator_name = "score_comparator"

        perturber = load_perturber(perturber_name=perturber_name)
        comparator = load_comparator(comparator_name=comparator_name)

        # ds = load_dataset("google/xquad", "xquad.en")
        ds = load_dataset("microsoft/ms_marco", "v1.1", split="test")
        passages = []
        for item in ds.select(range(5)):
            passages.extend(item["passages"]["passage_text"])
        corpus_documents = list(set(passages))

        # corpus_documents = list(set(ds["validation"]["context"]))[:100]
        encoder = Encoder(model_name="sentence-transformers")
        retriever = SBERTRetriever(encoder=encoder, corpus_documents=corpus_documents)
        explainer = GenericRetrieverExplainer(
            tokenizer=tokenizer,
            perturber=perturber,
            retriever=retriever,
            comparator=comparator,
        )

        explanation_dtos = explainer.explain(
            user_input=user_input,
            granularity=granularity,
            model_name=model_name,
        )

        system_responses = []
        generator_vis_es = []
        for explanation_dto in explanation_dtos:
            system_response = explanation_dto.output_text
            generator_vis = self.__visualize_explanations(
                user_input=explanation_dto.output_text,
                system_response=system_response,
                generator_explanations=explanation_dto,
                upper_percentile=int(upper_percentile),
                middle_percentile=int(middle_percentile),
                lower_percentile=int(lower_percentile),
            )

            system_responses.append(system_response)
            generator_vis_es.append(generator_vis)

        system_responses = "\n".join(system_responses)
        generator_vis_es = "\n".join(generator_vis_es)

        return system_responses, generator_vis_es

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
            with gr.Row(variant="compact"):
                explainer_name = gr.Radio(
                    label="Explainer",
                    choices=list(EXPLAINERS.keys()),
                    value=list(EXPLAINERS.keys())[0],
                    container=True,
                )
            with gr.Row(variant="compact"):
                upper_percentile = gr.Textbox(label="Upper", value="85", container=True)
                middle_percentile = gr.Textbox(
                    label="Middle", value="75", container=True
                )
                lower_percentile = gr.Textbox(label="Lower", value="10", container=True)

            with gr.Row(variant="compact"):
                model_name = gr.Radio(
                    label="Model",
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    container=True,
                )
            with gr.Row(variant="compact"):
                perturber_name = gr.Radio(
                    label="Perturber",
                    choices=list(PERTURBERS.keys()),
                    value=list(PERTURBERS.keys())[0],
                    container=True,
                )
            with gr.Row(variant="compact"):
                comparator_name = gr.Radio(
                    label="Comparator",
                    choices=list(COMPARATORS.keys()),
                    value=list(COMPARATORS.keys())[0],
                    container=True,
                )
        with gr.Row(variant="compact"):
            # passing "elem_id" to use a custom style for the component
            # in the CSS passed.
            submit_btn = gr.Button(
                value="🛠 Submit",
                variant="secondary",
                elem_id="button",
                interactive=True,
            )

        with gr.Row():
            generator_vis = gr.HTML(label="Explanations")

        with gr.Row():
            system_response = gr.Textbox(
                label="System Response",
                container=True,
                interactive=False,
            )

        return (
            user_input,
            system_response,
            granularity,
            upper_percentile,
            middle_percentile,
            lower_percentile,
            explainer_name,
            model_name,
            perturber_name,
            comparator_name,
            generator_vis,
            submit_btn,
        )

    def __visualize_explanations(
        self,
        user_input: str,
        system_response: Optional[str],
        generator_explanations: ExplanationDto,
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
            explanations=generator_explanations,
            output_from_explanations=user_input,
        )
