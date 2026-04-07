"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    image_url: HttpUrl | None = None


class AnalyzeResponse(BaseModel):
    prediction: Literal["fake", "real"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    fake_probability: float = Field(..., ge=0.0, le=1.0)
    evidence: list[dict[str, str | float]]
    explanation: str
    image_analysis: dict[str, bool | float]
