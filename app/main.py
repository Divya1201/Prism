from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.text_analysis import TextAnalysisService

app = FastAPI(title="Prism Fake News API")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)


class AnalyzeResponse(BaseModel):
    prediction: str
    confidence: float
    fake_probability: float


text_analysis_service: TextAnalysisService | None = None


@app.on_event("startup")
def load_model() -> None:
    global text_analysis_service
    try:
        text_analysis_service = TextAnalysisService()
    except FileNotFoundError:
        # Allow API boot without model; endpoint will provide a clear error.
        text_analysis_service = None


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
    if text_analysis_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train and save model before calling /analyze.",
        )

    prediction, confidence, fake_probability = text_analysis_service.predict(payload.text)
    return AnalyzeResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        fake_probability=round(fake_probability, 4),
    )
