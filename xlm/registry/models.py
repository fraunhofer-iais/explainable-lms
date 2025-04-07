from typing import Dict
import requests
from xlm.registry import DEFAULT_LMS_ENDPOINT


def get_models_from_lms(lms_endpoint: str = DEFAULT_LMS_ENDPOINT) -> Dict[str, str]:
    return get_models_based_on_condition(
        lms_endpoint=lms_endpoint,
        condition=lambda value: value["model_provider"] != "SentenceTransformers",
    )


def get_aleph_alpha_models_from_lms(
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
) -> Dict[str, str]:
    return get_models_based_on_condition(
        lms_endpoint=lms_endpoint,
        condition=lambda value: value["model_provider"] == "AlephAlpha",
    )


def get_models_based_on_condition(
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
    condition: callable = lambda value: True,
) -> Dict[str, str]:
    response = requests.get(f"{lms_endpoint}/available_models")
    if response.status_code == 200:
        result = response.json()
        models_dict = {key: key for key, value in result.items() if condition(value)}
        return models_dict
    else:
        raise ValueError(f"LMS is not connected!")


MODELS = get_models_from_lms()
