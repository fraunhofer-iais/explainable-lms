from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class FeatureImportance(BaseModel):
    feature: str
    score: float
    token_field: Optional[str] = None


class ExplanationDto(BaseModel):
    explanations: List[FeatureImportance]
    input_text: str
    output_text: Optional[str] = None


class ExplanationGranularity(str, Enum):
    WORD_LEVEL = "word_level_granularity"
    SENTENCE_LEVEL = "sentence_level_granularity"
    PARAGRAPH_LEVEL = "paragraph_level_granularity"
    PHRASE_LEVEL = "phrase_level_granularity"


class SimilarityMetric(Enum):
    COSINE = "cosine"
