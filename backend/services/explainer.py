"""Explainer service using LLM (HuggingFace API)"""

from __future__ import annotations

import os
import requests
from typing import Sequence, Iterable

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


class ExplainerService:
    def generate_explanation(
        self,
        input_text: str,
        prediction: str,
        retrieved_evidence: Sequence[str] | Iterable[str],
    ) -> dict[str, str]:

        # Format evidence nicely
        evidence_block = "\n".join(
            f"- {e}" for e in retrieved_evidence if e.strip()
        )

        prompt = f"""
You are an expert in misinformation analysis.

A piece of content has been classified as: {prediction}

Content:
{input_text}

Supporting evidence:
{evidence_block}

Explain:
1. Why this content may be misleading
2. What makes it suspicious
3. How the evidence supports this

Keep it clear and concise.
"""

        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": prompt},
                timeout=5
            )

            result = response.json()

            if isinstance(result, list):
                explanation = result[0].get("generated_text", "")
            else:
                explanation = str(result)

            explanation = explanation.strip()

            if not explanation:
                raise ValueError("Empty explanation")

        except Exception:
            # Fallback explanation
            explanation = (
                f"This content was classified as '{prediction}'. "
                f"Based on available evidence, it may contain misleading or incomplete information."
            )

        return {"explanation": explanation}
