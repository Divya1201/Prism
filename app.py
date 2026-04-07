from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.explainer import ExplainerService


app = FastAPI()
explainer_service = ExplainerService()


class AnalyzeRequest(BaseModel):
    input_text: str = Field(..., description="Claim text to analyze")
    retrieved_evidence: List[str] = Field(default_factory=list)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, str]:
    return explainer_service.generate_explanation(
        input_text=request.input_text,
        retrieved_evidence=request.retrieved_evidence,
    )
