from pydantic import BaseModel
from .bias_detector_response import BiasDetectorResponse
from .llm_response import LLMResponse

class AnalysisResult(BaseModel):
    bias_result: BiasDetectorResponse
    simplified_result: LLMResponse