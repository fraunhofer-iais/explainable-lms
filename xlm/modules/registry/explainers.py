from requests import Session
from xlm.explainer.aleph_alpha_explainer import AlephAlphaExplainer
from xlm.explainer.explainer import Explainer
from xlm.explainer.generic_explainer import GenericExplainer
from xlm.components.generator.llm_generator import LLMGenerator
from xlm.modules.registry import DEFAULT_LMS_ENDPOINT
from xlm.modules.registry.comparators import load_comparator
from xlm.modules.registry.perturbers import load_perturber
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer

EXPLAINERS = {
    "generic_explainer": "generic_explainer",
    "aleph_alpha_explainer": "aleph_alpha_explainer",
}


def load_explainer(
    explainer_name: str,
    perturber_name: str,
    model_name: str,
    comparator_name: str,
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
) -> Explainer:
    generator = LLMGenerator(session=Session(), endpoint=lms_endpoint)

    if explainer_name == "aleph_alpha_explainer":
        explainer = AlephAlphaExplainer(generator=generator)
    elif explainer_name == "generic_explainer":
        tokenizer = CustomTokenizer()
        perturber = load_perturber(perturber_name=perturber_name)
        comparator = load_comparator(
            comparator_name=comparator_name, model_name=model_name
        )
        explainer = GenericExplainer(
            tokenizer=tokenizer,
            perturber=perturber,
            generator=generator,
            comparator=comparator,
        )
    else:
        raise Exception(f"Invalid explainer passed: {explainer_name}")

    return explainer
