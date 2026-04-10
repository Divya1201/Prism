"""Service layer for misinformation detection using LLM (HuggingFace API)."""

from __future__ import annotations

import os
import requests
from backend.utils.preprocessing import preprocess_text

# ---------------------------
# CONFIG
# ---------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-small"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ---------------------------
# MAIN FUNCTION
# ---------------------------
def analyze_text(text: str, image_url: str | None = None) -> dict[str, str | float]:
    """Analyze text using LLM for misinformation classification."""

    processed_text = preprocess_text(text)

    # ---------------------------
    # PROMPT (VERY IMPORTANT)
    # ---------------------------
    prompt = f"""
You are an expert in misinformation detection.

Classify the following content into ONE of these categories:
- fabricated
- false_context
- manipulated
- imposter
- false_connection
- satire
- astroturfing
- sponsored
- unknown

Also provide a short explanation.

Text:
{processed_text}

Output format:
Category: <category>
Explanation: <reason>
"""

    # ---------------------------
    # API CALL
    # ---------------------------
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": prompt},
            timeout=5
        )

        if response.status_code != 200:
            raise RuntimeError(f"HF API error: {response.status_code}")

        result = response.json()

        # Handle different HF response formats
        if isinstance(result, list):
            generated_text = result[0].get("generated_text", "")
        else:
            generated_text = str(result)

        # ---------------------------
        # PARSE OUTPUT
        # ---------------------------
        prediction = "unknown"
        explanation = "No explanation generated."

        if "Category:" in generated_text:
            prediction = generated_text.split("Category:")[1].split("\n")[0].strip()

        if "Explanation:" in generated_text:
            explanation = generated_text.split("Explanation:")[1].strip()

        confidence = min(0.6 + len(prediction)/20, 0.9)

    except Exception as e:
        # ---------------------------
        # FALLBACK (VERY IMPORTANT)
        # ---------------------------
        prediction = "unknown"
        explanation = "Analysis failed or API unavailable."
        confidence = 0.5

    # ---------------------------
    # IMAGE ADJUSTMENT (OPTIONAL)
    # ---------------------------
    if image_url:
        confidence = max(confidence - 0.05, 0.0)

    return {
        "prediction": prediction,
        "confidence": round(min(confidence, 0.99), 2),
        "explanation": explanation,
    }
