from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .services.bias_detector import BiasDetector
from .config import get_settings
import logging

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

origins = settings.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptRequest(BaseModel):
    text: str

class BiasResponse(BaseModel):
    bias_score: float
    analysis: dict

bias_detector = BiasDetector(settings)

@app.get("/")
async def root():
    return {"message": "Welcome to the Bias Detection API"}

@app.post("/detect-bias", response_model=BiasResponse)
async def detect_bias(request: TranscriptRequest):
    bias_score, analysis = bias_detector.analyze_text(request.text)
    return BiasResponse(bias_score=bias_score, analysis=analysis)