"""FastAPI entrypoint for Prism backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
import logging
from backend.models.request_models import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

app = FastAPI(title="Prism API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pipeline = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    try:
        pipeline = AnalysisPipeline()
        logger.info("✅ Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        pipeline = None

@app.get("/")
def root():
    return {"message": "Prism API is running"}
    
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, object]:

    if pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline not initialized. Please check model setup.",
        )

    try:
        result = pipeline.run(
            text=payload.text,
            title=payload.title,
            url=str(payload.url) if payload.url else None,
            source=payload.source,
            author=payload.author,
            image_url=str(payload.image_url) if payload.image_url else None
        )

        logger.info(
            f"Prediction: {result.get('prediction')} | Confidence: {result.get('confidence')}"
        )

        return result

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis",
        )
