"""Core analysis pipeline orchestration."""

from __future__ import annotations
from typing import Any

from services.text_analysis import analyze_text
from services.retrieval import retrieve_evidence
from services.explainer import ExplainerService
from services.image_analysis import ImageAnalysisService

class AnalysisPipeline:
    """Coordinates baseline text analysis with retrieval/explanation placeholders."""
    def __init__(self):
        self.explainer = ExplainerService()
        self.image_service = ImageAnalysisService()
        
    def run(self, text: str, image_url: str | None = None):

        # 1. Validate input
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must not be empty")

        # 2. TEXT ANALYSIS (fake / real)
        text_result = analyze_text(cleaned, image_url)
        prediction = text_result["prediction"]
        confidence = text_result["confidence"]

        # 3. RETRIEVE EVIDENCE
        query = f"{cleaned} misinformation type: {prediction}"
        retrieval_result = retrieve_evidence(query)
        evidence_list = retrieval_result["evidence"]

        # Extract only text for explainer
        if not evidence_list:
            evidence_list = [{"text": "No supporting evidence found.", "score": 0.0}]
        evidence_texts = [item["text"] for item in evidence_list]
            
        # 4. GENERATE EXPLANATION (LLM)
        try:
            explanation_result = self.explainer.generate_explanation(
                input_text=f"{cleaned}\nPredicted category: {prediction}",
                retrieved_evidence=evidence_texts
            )
            explanation_llm = explanation_result.get("explanation", "No explanation generated.")
        except Exception:
            explanation_llm = "Explanation generation failed. Showing retrieved evidence instead." 

        # 5. MERGE EXPLANATIONS
        final_explanation = (
            f"{text_result['explanation']}\n\n"
            f"Supporting analysis:\n{explanation_llm}"
        )
        
        # 6. IMAGE ANALYSIS (optional)
        if image_url:
            try:
                image_analysis = self.image_service.analyze_image_url(image_url)
                image_analysis["enabled"] = True
            except Exception:
                image_analysis = {"enabled": False, "error": "Image analysis failed"}
        else:
            image_analysis = {"enabled": False}

        # 7. FINAL RESPONSE
        return {
            "prediction": prediction,
            "confidence": confidence,
            "evidence": evidence_list,
            "explanation": final_explanation,
            "image_analysis": image_analysis,
        }
