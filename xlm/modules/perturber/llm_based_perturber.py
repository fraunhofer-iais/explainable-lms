from typing import List
from requests import Session
from xlm.components.generator.llm_generator import LLMGenerator
from xlm.modules.perturber.perturber import Perturber
from xlm.modules.registry import DEFAULT_LMS_ENDPOINT


class LLMBasedPerturber(Perturber):
    def __init__(
        self,
        template_path: str,
        model_name: str = "gpt-3.5-turbo",
        endpoint: str = DEFAULT_LMS_ENDPOINT,
    ):
        self.__model_name = model_name
        self.__generator = LLMGenerator(
            session=Session(), endpoint=endpoint, model_name=model_name
        )
        self.__prompt_template = self.__read_prompt_template(path=template_path)

    def perturb(self, text: str, features: List[str]) -> List[str]:
        perturbations = []
        prompts = []
        for feature in features:
            prompt = self.__prompt_template.format(sentence=feature)
            prompts.append(prompt)

        responses = self.__generator.generate(texts=prompts)

        for response, feature in zip(responses, features):
            perturbations.append(text.replace(feature, response).strip())
        return perturbations

    def __read_prompt_template(self, path: str) -> str:
        with open(path, "r") as f:
            data = f.read()

        return data.strip()
