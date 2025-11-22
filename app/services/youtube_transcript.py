import logging
from ..config import get_settings
from urllib.parse import urlparse, parse_qs
import subprocess
import os

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

        yt_dlp_captions = self.get_youtube_captions(video_id, cookies_path=self.settings.YT_DLP_COOKIES_PATH)
        logger.info(f"Transcript fetched successfully via yt-dlp: {yt_dlp_captions[:100]}...")

        transcript_text = self.srt_to_paragraph(yt_dlp_captions)
        logger.info(f"Transcript after srt to paragraph conversion: {transcript_text[:100]}...")
        
        return transcript_text
    

    def get_youtube_captions(self, video_id: str, cookies_path: str = "cookies.txt") -> str:
        """
        Download and return YouTube auto-sub captions for a given video ID using yt-dlp.
        Returns the raw caption file content exactly as provided (no formatting).
        Automatically deletes the generated caption file after reading.
        """

        url = f"https://www.youtube.com/watch?v={video_id}"

        # Run yt-dlp command to download auto subtitles
        command = [
            "yt-dlp",
            "--cookies", cookies_path,
            "--write-auto-sub",
            "--convert-subs", "srt",
            "--sub-lang", "en",
            "--skip-download",
            url,
        ]



        try:
            logger.info(f"Running yt-dlp command: {' '.join(command)}")
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            logger.error(f"yt-dlp failed to download captions for video ID: {video_id}")
            return ""

        caption_file = None

        for file in os.listdir("."):
            if file.endswith(".srt") and video_id in file:
                caption_file = file
                break

        if not caption_file:
            for file in os.listdir("."):
                if file.endswith(".srt"):
                    caption_file = file
                    break

        if not caption_file:
            return ""

        with open(caption_file, "r", encoding="utf-8") as f:
            content = f.read()

        try:
            os.remove(caption_file)
        except OSError:
            pass

        return content
    
    def srt_to_paragraph(self,srt_text: str) -> str:
        """
        Convert raw .srt subtitle text (with timestamps + line numbers) into a clean paragraph.
        Removes numbering and timestamps, merges repeating lines, and returns a single paragraph.
        """

        lines = srt_text.splitlines()
        cleaned_lines = []
        last_line = ""

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.isdigit():
                continue

            if "-->" in line:
                continue

            if line != last_line:
                cleaned_lines.append(line)
                last_line = line

        paragraph = " ".join(cleaned_lines)
        return paragraph

youtube_transcript = YoutubeTranscriptService(get_settings())