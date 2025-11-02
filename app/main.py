from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis
from pydantic import BaseModel
from .services.bias_detector import BiasDetectorService
from .services.youtube_transcript import YoutubeTranscriptService
from .services.llm_response import LLMResponseService
from .interfaces.bias_detector_response import BiasDetectorResponse
from .interfaces.analysis_result import AnalysisResult
from .config import get_settings
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(
    title="Bias Detection API",
    description="API for detecting bias in YouTube transcripts",
    version="1.0.0"
)

r = aioredis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

origins = settings.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    url: str

bias_detector = BiasDetectorService(settings)
youtube_transcript_generator = YoutubeTranscriptService(settings)
llm_response = LLMResponseService(settings)

@app.get("/")
async def root():
    return {"message": "Welcome to the Bias Detection API"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(request: AnalysisRequest):

    cache_key = f"video:{request.url}"
    try:
        cached_data = await r.get(cache_key)
        if cached_data:
            logger.info("Cache hit. Returning cached analysis result.")
            return AnalysisResult(**json.loads(cached_data))
        
        transcript = youtube_transcript_generator.generate_transcript(request.url)
        bias_score, analysis = bias_detector.analyze_text(transcript)
        understandable_response = llm_response.generate_understandable_response(
            BiasDetectorResponse(bias_score=bias_score, analysis=analysis),
            transcript
        )

        cache_data = {
            "bias_result": {
                "bias_score": bias_score,
                "analysis": analysis
            },
            "simplified_result": understandable_response.dict()
        }
        
        logger.info("Caching analysis result.")
        
        await r.setex(name=cache_key, time=3600, value=json.dumps(cache_data))
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")

    return AnalysisResult(
        bias_result=BiasDetectorResponse(bias_score=bias_score, analysis=analysis),
        simplified_result=understandable_response
    )