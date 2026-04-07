"""Pydantic models for request and response payloads."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AnalyzeRequest(BaseModel):
    """Incoming payload for misinformation analysis."""

    text: str = Field(..., min_length=1, description="Text content to analyze")
    image_url: HttpUrl | None = Field(
        default=None,
        description="Optional URL to an image related to the claim",
    )


class AnalyzeResponse(BaseModel):
    """API response containing model prediction details."""

    prediction: Literal["fake", "real"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str
