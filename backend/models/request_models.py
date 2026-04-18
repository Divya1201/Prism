"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, HttpUrl

MisinformationLabel = Literal[
    "fabricated",
    "false_context",
    "manipulated",
    "imposter",
    "false_connection",
    "satire",
    "astroturfing",
    "sponsored",
    "not_misinformation"
    "unverified",
]

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    source: Optional[str] = None
    author: Optional[str] = None

    image_url: Optional[HttpUrl] = None

class EvidenceItem(BaseModel):
    text: str
    source: str
    score: float = Field(..., ge=0.0, le=1.0)
    provider: str = "unknown"
    
class AnalyzeResponse(BaseModel):
    prediction: MisinformationLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    key_claims: list[str]
    evidence: list[EvidenceItem]
    explanation: str
    image_analysis: dict[str, object]
