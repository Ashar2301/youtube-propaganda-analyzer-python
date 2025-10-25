# YouTube Bias Detection API

This is a FastAPI-based service that analyzes YouTube transcripts for bias using machine learning models from Hugging Face.

## Setup

1. Ensure you have Python 3.8+ installed
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the API

To start the API server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /detect-bias

Analyze text for bias.

Request body:
```json
{
    "text": "Your transcript text here"
}
```

Response:
```json
{
    "bias_score": 0.5,
    "analysis": {
        "raw_scores": [],
        "detected_biases": [],
        "confidence": 0.0,
        "text_length": 123
    }
}
```

## Notes

- The current implementation uses a placeholder model. You should replace it with your specific bias detection model from Hugging Face.
- Configure the model path in `app/services/bias_detector.py`