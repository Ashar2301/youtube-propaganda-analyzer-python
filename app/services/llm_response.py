import logging
from ..config import get_settings
from google import genai
from google.genai import types
from ..interfaces.bias_detector_response import BiasDetectorResponse
from ..interfaces.llm_response import LLMResponse
import json

logger = logging.getLogger(__name__)

class LLMResponseService:
    """
    Loads the Hugging Face model components (Tokenizer and TF Model)
    and handles bias detection inference.
    """
    def __init__(self, settings):
        self.settings = settings

    def generate_understandable_response(self, bias_analysis_result: BiasDetectorResponse, transcript_text: str) -> LLMResponse:
        """
        Generate a human-understandable response using Gemini LLM.
        
        Args:
            bias_analysis_result: The bias analysis results from the model
            transcript_text: The transcript text that was analyzed
            
        Returns:
            LLMResponse: A structured response containing explanation and tags
        """
        client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
        logger.info("Generating understandable response using Gemini LLM.")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""Data:
                Bias score: {bias_analysis_result.bias_score}
                Label: {bias_analysis_result.analysis.top_prediction_label}
                Confidence: {bias_analysis_result.analysis.top_prediction_confidence}
                Transcript excerpt: "{transcript_text[:300]}...\"""",
            config=types.GenerateContentConfig(
                response_mime_type= "application/json",
                system_instruction= """You are an AI assistant that analyzes bias detection results from videos.
Always respond only in valid JSON, never include explanations or extra text.
Output must include exactly these keys:

"explanation": a short, clear summary (2 to 3 sentences) describing whether the video contains propaganda or bias and why.

"tags": an array of 2 to 5 lowercase strings capturing relevant concepts or techniques (e.g., ["loaded language", "framing", "political bias"])."""
            )
        )
        logger.info(f"LLM response received: {response.text}")
        
        json_response = json.loads(response.text)
        return LLMResponse(**json_response)

llm_response = LLMResponseService(get_settings())