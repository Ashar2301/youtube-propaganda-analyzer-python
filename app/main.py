from fastapi import FastAPI
from pydantic import BaseModel
from .services.bias_detector import BiasDetector

app = FastAPI(title="Bias Detection API",
             description="API for detecting bias in YouTube transcripts",
             version="1.0.0")

class TranscriptRequest(BaseModel):
    text: str

class BiasResponse(BaseModel):
    bias_score: float
    analysis: dict

bias_detector = BiasDetector()

@app.get("/")
async def root():
    return {"message": "Welcome to the Bias Detection API"}

@app.post("/detect-bias", response_model=BiasResponse)
async def detect_bias(request: TranscriptRequest):
    bias_score, analysis = bias_detector.analyze_text(request.text)
    return BiasResponse(bias_score=bias_score, analysis=analysis)