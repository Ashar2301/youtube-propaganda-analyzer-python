from pydantic import BaseModel
from .BiasDetector import BiasResponse
from .LLMResponse import LLMResponse

class AnalysisResult(BaseModel):
    bias_result: BiasResponse
    simplifiedResult: LLMResponse