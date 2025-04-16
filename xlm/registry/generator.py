from xlm.components.generator.llm_generator import LLMGenerator
from xlm.registry import DEFAULT_LMS_ENDPOINT


def load_generator(
    generator_model_name: str,
    split_lines: bool,
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
):
    return LLMGenerator(
        endpoint=lms_endpoint, model_name=generator_model_name, split_lines=split_lines
    )
