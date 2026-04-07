"""Route handlers for analysis endpoints."""

from fastapi import APIRouter

from app.models.request_models import AnalyzeRequest, AnalyzeResponse
from app.services.text_analysis import analyze_text

router = APIRouter(prefix="", tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_payload(payload: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze payload text and return fake/real prediction with confidence."""
    result = analyze_text(text=payload.text, image_url=str(payload.image_url) if payload.image_url else None)
    return AnalyzeResponse(**result)
