"""FastAPI entrypoint for Prism ML backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from backend.models.request_models import AnalyzeRequest, AnalyzeResponse
from backend.pipeline import AnalysisPipeline

app = FastAPI(title="Prism ML API")

pipeline: AnalysisPipeline | None = None


@app.on_event("startup")
def startup() -> None:
    global pipeline
    try:
        pipeline = AnalysisPipeline()
    except FileNotFoundError:
        pipeline = None


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model before calling /analyze.")

    result = pipeline.run(text=payload.text, image_url=str(payload.image_url) if payload.image_url else None)
    return AnalyzeResponse(**result)
