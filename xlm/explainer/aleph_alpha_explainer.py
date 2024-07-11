import os
from typing import Optional

import numpy as np
from aleph_alpha_client import (
    Client,
    Prompt,
    CompletionRequest,
    ExplanationRequest,
    TargetGranularity,
)

from xlm.dto.dto import (
    ExplanationDto,
    ExplanationGranularity,
    FeatureImportance,
)
from xlm.explainer.explainer import Explainer
from xlm.generator.generator import Generator
from xlm.registry.models import get_aleph_alpha_models_from_lms


class AlephAlphaExplainer(Explainer):
    def __init__(self, generator: Generator):
        self.__generator = generator
        self.__client = self.__get_client()

    def __get_client(self) -> Optional[Client]:
        try:
            return Client(token=os.getenv("ALEPH_ALPHA_TOKEN"))
        except TypeError as e:
            raise Exception(
                "Please set your Aleph Alpha token. Use the variable "
                "ALEPH_ALPHA_TOKEN."
            )

    def explain(
            self,
            user_input: str,
            granularity: ExplanationGranularity,
            model_name: Optional[str] = None,
            normalize: Optional[bool] = True,
            system_response: Optional[str] = None,
            maximum_tokens: Optional[int] = 100
    ) -> ExplanationDto:
        self.__validate_model_name(model_name)
        reference_response = (
            self.__generator.generate(texts=[user_input], model_name=model_name)[0]
            if not system_response
            else system_response
        )
        prompt = Prompt.from_text(user_input)
        params = {
            "prompt": prompt,
            "maximum_tokens": maximum_tokens,
        }
        request = CompletionRequest(**params)
        response = self.__client.complete(request=request, model=model_name)
        completion = response.completions[0].completion
        exp_req = ExplanationRequest(
            prompt=prompt,
            target=completion,
            control_factor=0.1,
            control_log_additive=False,
            target_granularity=TargetGranularity.Complete,
            prompt_granularity=self.__get_prompt_granularity(granularity),
            normalize=normalize,
        )
        raw_explanations = self.__client.explain(request=exp_req, model=model_name)
        explanation_dto = self.__format_explanations(
            raw_explanations=raw_explanations,
            user_input=user_input,
            reference_response=reference_response,
        )
        return explanation_dto

    def __validate_model_name(self, model_name: str):
        available_aleph_alpha_models = self.__get_aleph_alpha_models_from_lms()
        assert model_name in available_aleph_alpha_models, (
            f"Aleph Alpha Explainer is only supported for AA models. Please "
            f"use any of the following AA models: {available_aleph_alpha_models}"
        )

    def __get_aleph_alpha_models_from_lms(self) -> dict[str, str]:
        return get_aleph_alpha_models_from_lms()


    def __get_prompt_granularity(self, granularity: ExplanationGranularity):
        return {
            ExplanationGranularity.WORD_LEVEL: "word",
            ExplanationGranularity.SENTENCE_LEVEL: "sentence",
            ExplanationGranularity.PARAGRAPH_LEVEL: "paragraph",
        }[granularity]

    def __format_explanations(
            self, raw_explanations, user_input, reference_response
    ) -> ExplanationDto:
        explanations = raw_explanations.explanations[0].items[0].scores
        scores = []
        features = []
        for item in explanations:
            start = item.start
            end = item.start + item.length
            features.append(user_input[start:end])
            scores.append(np.round(item.score, decimals=2))
        explanations = [
            FeatureImportance(feature=feature, score=score)
            for feature, score in zip(features, scores)
        ]
        return ExplanationDto(
            explanations=explanations,
            input_text=user_input,
            output_text=reference_response,
        )

