from __future__ import annotations

import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ExplainerService:
    def generate_explanation(
        self,
        input_text: str,
        prediction: str,
        retrieved_evidence,
    ) -> dict[str, str]:

        # Limit input size (important for API)
        input_text = input_text[:2000]

        # Format evidence (top 5 only)
        evidence_block = "\n".join(
            f"- {e}" for e in list(retrieved_evidence)[:5] if e
        )

        prompt = f"""
You are an expert in misinformation detection.

Content:
{input_text}

Predicted category: {prediction}

Evidence:
{evidence_block}

Explain clearly:
1. Why this content is misleading
2. What signals indicate misinformation
3. How evidence supports this

Keep it concise and factual.
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )

            explanation = response.choices[0].message.content.strip()

        except Exception:
            explanation = (
                f"This content is classified as '{prediction}'. "
                f"Based on available evidence, it may be misleading."
            )

        return {"explanation": explanation}
