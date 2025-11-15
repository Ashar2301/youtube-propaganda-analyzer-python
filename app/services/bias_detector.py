from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification
import logging
from typing import Dict, Any, List, Tuple
from ..config import get_settings

logger = logging.getLogger(__name__)

class BiasDetectorService:
    """
    Loads the Hugging Face model components (Tokenizer and TF Model)
    and handles bias detection inference.
    """
    def __init__(self, settings):
        self.settings = settings
        self.tokenizer = None
        self.model = None
        self.classifier = None 

    def load_model(self):
        """Initializes the tokenizer, model, and the Hugging Face pipeline."""
        if self.classifier is None:
            model_name = self.settings.MODEL_NAME
            cache_dir = self.settings.MODEL_CACHE_DIR
            
            logger.info(f"Loading Tokenizer and TF Model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                self.model = TFAutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    from_pt=False
                )

                self.classifier = pipeline(
                    'text-classification', 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    framework="tf",
                    device=-1,       # Use -1 for CPU, 0 for first GPU
                    return_all_scores=True
                )
                logger.info("Bias Detection Pipeline loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading model components: {e}")
                raise RuntimeError(f"Could not load model '{model_name}'. Check TensorFlow installation.") from e

    def analyze_text(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze text for bias using the loaded pipeline.
        
        Returns:
            tuple: (overall_bias_score, detailed_analysis)
        """
        if self.classifier is None:
             self.load_model()

        logger.info(f"Analyzing text for bias. Text length: {len(text)}")

        results: List[List[Dict[str, Any]]] = self.classifier(text)
        
        logger.info("No issues in classifier execution.")
        
        raw_scores = results[0] if results else []

        
        bias_score = 0.0
        top_bias_category = "Non-Bias"
        confidence = 0.0

        for score_info in raw_scores:
            label = score_info['label']
            score = score_info['score']
            
            if label.lower() == 'biased':
                bias_score = round(score, 4)
                confidence = bias_score
                top_bias_category = label
                break 

            if score > confidence:
                 confidence = round(score, 4)
                 top_bias_category = label

        
        analysis = {
            "overall_bias_score": bias_score,
            "top_prediction_label": top_bias_category,
            "top_prediction_confidence": confidence,
            "raw_scores": raw_scores
        }

        logger.info(f"Bias analysis completed. Score: {bias_score}, Top Label: {top_bias_category}, Confidence: {confidence}")
        
        return bias_score, analysis

bias_detector = BiasDetectorService(get_settings())