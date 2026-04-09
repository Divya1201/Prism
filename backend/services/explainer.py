from __future__ import annotations

import os
import requests
from typing import Iterable, Sequence

# -----------------------------
# CONFIG
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


# -----------------------------
# EXPLAINER SERVICE
# -----------------------------
class ExplainerService:
    def generate_explanation(
        self,
        input_text: str,
        retrieved_evidence: Sequence[str] | Iterable[str],
    ) -> dict[str, str]:

        # Prepare evidence
        evidence_lines = [
            f"- {line.strip()}"
            for line in retrieved_evidence
            if line and line.strip()
        ]

        evidence_block = (
            "\n".join(evidence_lines)
            if evidence_lines
            else "- No evidence retrieved."
        )

        # Prompt
        prompt = (
            "Explain why this claim may be misleading.\n\n"
            f"Claim: {input_text.strip()}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Answer briefly:"
        )

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt},
                timeout=10
            )

            result = response.json()

            if isinstance(result, list):
                explanation = result[0]["generated_text"]
            else:
                explanation = "Explanation unavailable."

        except Exception:
            explanation = "Error generating explanation."

        return {"explanation": explanation}
