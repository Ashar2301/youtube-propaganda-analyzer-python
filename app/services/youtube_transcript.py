import logging
from ..config import get_settings
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

class YoutubeTranscriptService:
    """
    Loads the Hugging Face model components (Tokenizer and TF Model)
    and handles bias detection inference.
    """
    def __init__(self, settings):
        self.settings = settings

    def extract_video_id(self,url:str):
        return parse_qs(urlparse(url).query).get("v", [None])[0] 

    def generate_transcript(self, url: str) -> str:
        """
        Analyze text for bias using the loaded pipeline.
        
        Returns:
            tuple: (overall_bias_score, detailed_analysis)
        """
        video_id = self.extract_video_id(url)
        logger.info(f"Fetching transcript for video ID: {video_id}")

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        transcript_text = ""
        for snippet in transcript:
            transcript_text += " " + snippet.text
        
        logger.info(f"Transcript fetched successfully: {transcript_text[:100]}...")  
        
        return transcript_text;

youtube_transcript = YoutubeTranscriptService(get_settings())