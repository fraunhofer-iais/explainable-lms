import pytest
from xlm.dto.dto import ExplanationGranularity
from xlm.modules.perturber.leave_one_out_perturber import LeaveOneOutPerturber
from xlm.modules.perturber.llm_based_perturber import LLMBasedPerturber
from xlm.modules.perturber.random_word_perturber import RandomWordPerturber
from xlm.modules.perturber.reorder_perturber import ReorderPerturber
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer


@pytest.fixture
def tokenizer():
    return CustomTokenizer()


@pytest.fixture
def text():
    return """\
    Published at a time of rising demand for German-language publications, Luther's version quickly became a \
popular and influential Bible translation. As such, it made a significant contribution to the evolution of the \
German language and literature. Furnished with notes and prefaces by Luther, and with woodcuts by Lucas Cranach \
that contained anti-papal imagery, it played a major role in the spread of Luther's doctrine throughout Germany. \
The Luther Bible influenced other vernacular translations, such as William Tyndale's English Bible (1525 forward),\
 a precursor of the King James Bible.
 """


@pytest.fixture
def base_path():
    return "data/prompt_templates/"


def run_llm_perturber_based_on_template_path(
    template_path: str, text: str, tokenizer: CustomTokenizer
):
    perturber = LLMBasedPerturber(template_path=template_path)
    perturbations = perturber.perturb(
        text=text,
        features=tokenizer.tokenize(
            text=text, granularity=ExplanationGranularity.SENTENCE_LEVEL
        ),
    )
    return perturbations


def assert_pertubations(perturbations: list[str], text: str):
    assert len(perturbations) == 4
    assert isinstance(perturbations, list)
    assert isinstance(perturbations[0], str)
    for perturbed_text in perturbations:
        assert perturbed_text != text


def test_llm_based_antonym_perturber(base_path, text, tokenizer):
    filename = "llm_based_antonym_perturber_template.txt"
    perturbations = run_llm_perturber_based_on_template_path(
        base_path + filename, text, tokenizer
    )
    assert_pertubations(perturbations, text)


def test_llm_based_synonym_perturber(base_path, text, tokenizer):
    filename = "llm_based_synonym_perturber_template.txt"
    perturbations = run_llm_perturber_based_on_template_path(
        base_path + filename, text, tokenizer
    )
    assert_pertubations(perturbations, text)


def test_llm_based_entity_perturber(base_path, text, tokenizer):
    filename = "llm_based_entity_perturber_template.txt"
    perturbations = run_llm_perturber_based_on_template_path(
        base_path + filename, text, tokenizer
    )
    assert_pertubations(perturbations, text)


@pytest.fixture(
    params=[
        {
            "text": "This is the first sentence. This is the second sentence.",
            "features": ["This is the first sentence.", "This is the second sentence."],
        }
    ]
)
def input_data(request):
    return request.param


def assert_perturbations(original_text, perturbations, features):
    assert perturbations != original_text

    for feature, perturbation in zip(features, perturbations):
        assert feature not in perturbation

    assert len(perturbations) == len(features)


@pytest.fixture
def leave_one_out_perturber():
    return LeaveOneOutPerturber()


@pytest.fixture
def random_word_perturber():
    return RandomWordPerturber()


@pytest.fixture
def reorder_perturber():
    return ReorderPerturber()


def test_leave_one_out_perturber(leave_one_out_perturber, input_data):
    text = input_data["text"]
    features = input_data["features"]
    perturbations = leave_one_out_perturber.perturb(text=text, features=features)
    assert_perturbations(text, perturbations, features)


def test_reorder_perturber(reorder_perturber, input_data):
    text = input_data["text"]
    features = input_data["features"]
    perturbations = reorder_perturber.perturb(text=text, features=features)
    assert_perturbations(text, perturbations, features)


def test_random_word_perturber(random_word_perturber, input_data):
    text = input_data["text"]
    features = input_data["features"]
    perturbations = random_word_perturber.perturb(text=text, features=features)
    assert_perturbations(text, perturbations, features)
