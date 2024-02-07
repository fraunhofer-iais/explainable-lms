from perturber.leave_one_out_perturber import LeaveOneOutPerturber
from perturber.llm_based_perturber import LLMBasedPerturber
from perturber.random_word_perturber import RandomWordPerturber
from perturber.reorder_perturber import ReorderPerturber

PERTURBERS = {
    "leave_one_out": LeaveOneOutPerturber(),
    "random_word_perturber": RandomWordPerturber(),
    "reorder_perturber": ReorderPerturber(),
    "antonym_perturber": LLMBasedPerturber(
        template_path="data/prompt_templates"
        "/llm_based_antonym_perturber_template.txt"
    ),
    "synonym_perturber": LLMBasedPerturber(
        template_path="data/prompt_templates"
        "/llm_based_synonym_perturber_template.txt"
    ),
    "entity_perturber": LLMBasedPerturber(
        template_path="data/prompt_templates"
        "/llm_based_entity_perturber_template.txt"
    ),
}


def load_perturber(perturber_name: str):
    if perturber_name not in PERTURBERS.keys():
        raise Exception(
            f"The entered perturber name is not found! Available "
            f"perturbers are: {list(PERTURBERS.keys())}"
        )

    return PERTURBERS.get(perturber_name)
