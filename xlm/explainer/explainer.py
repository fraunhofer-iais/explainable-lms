from abc import ABC, abstractmethod
from typing import Optional

from xlm.dto.dto import ExplanationGranularity, ExplanationDto


class Explainer(ABC):
    @abstractmethod
    def explain(
        self,
        user_input: str,
        granularity: ExplanationGranularity,
        model_name: Optional[str] = None,
        do_normalize_comparator_scores: Optional[bool] = True,
        system_response: Optional[str] = None,
    ) -> ExplanationDto: ...
