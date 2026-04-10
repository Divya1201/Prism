"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    source: Optional[str] = None
    author: Optional[str] = None

    image_url: Optional[HttpUrl] = None


class AnalyzeResponse(BaseModel):
    prediction: Literal[
        "fabricated",
        "false_context",
        "manipulated",
        "imposter",
        "false_connection",
        "satire",
        "astroturfing",
        "sponsored",
        "unknown",
    ]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[dict[str, str | float]]
    explanation: str
    image_analysis: dict[str, bool | float]
