"""FastAPI entrypoint for Prism backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from backend.models.request_models import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

app = FastAPI(title="Prism API")
pipeline = AnalysisPipeline()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, object]:
    try:
        return pipeline.run(payload.text, str(payload.image_url) if payload.image_url else None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
