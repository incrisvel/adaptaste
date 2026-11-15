from typing import Literal

from pydantic import BaseModel


class ReviewAnalysis(BaseModel):
    review_id: str
    sentiment: Literal["positive", "negative"]
    sentiment_confidence: float
    aspects: list["AspectAnalysis"]


class AspectAnalysis(BaseModel):
    aspect: str
    polarity: Literal["positive", "negative", "neutral"]
    evidence: str
