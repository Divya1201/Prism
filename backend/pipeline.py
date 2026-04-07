"""Core analysis pipeline orchestration."""

from __future__ import annotations

from typing import Any


class AnalysisPipeline:
    """Coordinates text, retrieval, explanation, and optional image analysis."""

    def __init__(self) -> None:
        from backend.services.text_analysis import TextAnalysisService

        self.text_analysis = TextAnalysisService()

        from backend.services.explainer import ExplainerService
        from backend.services.image_analysis import ImageAnalysisService
        from backend.services.retrieval import retrieve_evidence

        self.explainer = ExplainerService()
        self.image_analysis = ImageAnalysisService()
        self.retrieve_evidence = retrieve_evidence

    def run(self, text: str, image_url: str | None = None) -> dict[str, Any]:
        prediction, confidence, fake_probability = self.text_analysis.predict(text)

        evidence_payload = self.retrieve_evidence(text, top_k=3)
        evidence_text = [item["text"] for item in evidence_payload["evidence"]]

        explanation_payload = self.explainer.generate_explanation(
            input_text=text,
            retrieved_evidence=evidence_text,
        )

        image_payload = (
            self.image_analysis.analyze_image_url(image_url)
            if image_url
            else {"image_similarity_score": 0.0, "possible_reuse": False}
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "fake_probability": round(fake_probability, 4),
            "evidence": evidence_payload["evidence"],
            "explanation": explanation_payload["explanation"],
            "image_analysis": image_payload,
        }
