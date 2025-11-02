from pydantic import BaseModel
from typing import List

class RawScore(BaseModel):
    """
    Represents a single score with its label from the bias detection model.
    """
    label: str
    score: float

class Analysis(BaseModel):
    """
    Detailed analysis results from the bias detection model.
    """
    overall_bias_score: float
    top_prediction_label: str
    top_prediction_confidence: float
    raw_scores: List[RawScore]

class BiasDetectorResponse(BaseModel):
    """
    Complete response from the bias detection analysis.
    """
    bias_score: float
    analysis: Analysis