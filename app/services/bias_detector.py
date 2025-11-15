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

        # Split text into chunks of max 2000 characters
        max_chunk_size = 2000
        text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        logger.info(f"Text split into {len(text_chunks)} chunks for analysis.") 
        analysis_array = []

        for text_chunk in text_chunks:
            result: List[List[Dict[str, Any]]] = self.classifier(text_chunk)
            raw_score = result[0] if result else []
        
            bias_score = 0.0
            top_bias_category = "Non-Bias"
            confidence = 0.0

            for score_info in raw_score:
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
                "raw_scores": raw_score
            }

            logger.info(f"Chunk analysis completed. Score: {bias_score}, Top Label: {top_bias_category}, Confidence: {confidence}")

            analysis_array.append(analysis)


        averaged_bias_score = self.compute_average_analysis(analysis_array)

        logger.info(f"Bias analysis completed. {averaged_bias_score}")
        
        return averaged_bias_score
    
    def compute_average_analysis(self,analysis_array) -> Tuple[float, Dict[str, Any]]:
        if not analysis_array:
            return{ 
                "bias_score": 0.0,
                "analysis":{
                    "overall_bias_score": 0.0,
                    "top_prediction_label": None,
                    "top_prediction_confidence": 0.0,
                    "raw_scores": []
                }
            }

        total_bias_score = 0.0
        total_confidence = 0.0

        # raw_scores is a list of dicts; we must average each label's score separately
        score_sums = {}
        score_counts = {}

        for entry in analysis_array:
            total_bias_score += entry["overall_bias_score"]
            total_confidence += entry["top_prediction_confidence"]

            for score_item in entry["raw_scores"]:
                label = score_item["label"]
                score = score_item["score"]

                score_sums[label] = score_sums.get(label, 0.0) + score
                score_counts[label] = score_counts.get(label, 0) + 1

        avg_bias_score = total_bias_score / len(analysis_array)
        avg_confidence = total_confidence / len(analysis_array)

        avg_raw_scores = [
            {"label": label, "score": score_sums[label] / score_counts[label]}
            for label in score_sums
        ]

        # Determine label with highest averaged score
        top_prediction_label = max(avg_raw_scores, key=lambda x: x["score"])["label"]

        return {
            "bias_score": avg_bias_score,
            "analysis": {
                "overall_bias_score": avg_bias_score,
                "top_prediction_label": top_prediction_label,
                "top_prediction_confidence": avg_confidence,
                "raw_scores": avg_raw_scores
            }
        }


bias_detector = BiasDetectorService(get_settings())