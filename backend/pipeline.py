"""Core analysis pipeline orchestration."""

from __future__ import annotations

from backend.services.text_analysis import analyze_text
from backend.services.retrieval import retrieve_evidence
from backend.services.explainer import ExplainerService
from backend.services.image_analysis import ImageAnalysisService

class AnalysisPipeline:
    """Coordinates text analysis, retrieval, explanation, and image analysis."""
    def __init__(self):
        self.explainer = ExplainerService()
        self.image_service = ImageAnalysisService()

    def build_query(text: str, title: str | None):
        if title and len(title) > 10:
            return f"{title} news"

    sentences = text.split(".")
    for s in sentences:
        s = s.strip()
        if len(s) > 40:
            return f"{s} news"

    return text[:120]
    
    def run(
        self,
        text: str,
        title: str | None = None,
        url: str | None = None,
        source: str | None = None,
        author: str | None = None,
        image_url: str | None = None,
    ):

        # 1. BUILD CONTEXT

        if not text or not text.strip():
            raise ValueError("text must not be empty")
            
        clean_text = text.strip()

        # limit size 
        clean_text = clean_text[:2000]

        context_parts = [clean_text]

        if title:
            context_parts.append(f"Title: {title}")

        if source:
            context_parts.append(f"Source: {source}")

        if author:
            context_parts.append(f"Author: {author}")

        if url:
            context_parts.append(f"URL: {url}")

        cleaned = "\n".join(context_parts).strip()

        if not cleaned:
            raise ValueError("text must not be empty")

        # 2. TEXT ANALYSIS (fake / real)
        
        text_result = analyze_text(cleaned, image_url)
        prediction = text_result.get("prediction", "unknown")
        confidence = text_result.get("confidence", 0.5)

        if confidence < 0.3:
            prediction = "uncertain"

        # 3. RETRIEVE EVIDENCE
        
        query = self.build_query(text, title)
        retrieval_result = retrieve_evidence(query)
        evidence_list = retrieval_result.get("evidence", [])

        # Extract only text for explainer
        if not evidence_list:
            evidence_list = [{
                "text": "No external evidence found. The analysis is based on linguistic patterns.",
                "source": "system",
                "score": 0.0
            }]
        evidence_texts = [item["text"] for item in evidence_list]
            
        # 4. GENERATE EXPLANATION (LLM)
        try:
            explanation_result = self.explainer.generate_explanation(
                input_text=cleaned,
                prediction=prediction,
                retrieved_evidence=evidence_texts
            )

            explanation_llm = explanation_result.get(
                "explanation",
                "No explanation generated."
            )

        except Exception:
            explanation_llm = (
                "Explanation generation failed. Showing retrieved evidence instead."
            )

        # 5. MERGE EXPLANATIONS
        final_explanation = explanation_llm.strip()
        
        # 6. IMAGE ANALYSIS 
        image_analysis = {"enabled": False}
        if image_url:
            try:
                image_analysis = self.image_service.analyze_image_url(image_url)
            except Exception:
                image_analysis = {"enabled": False, "error": "Image analysis failed"}
            
        # MERGE iMAGE INSIGHT
        if image_analysis.get("enabled") and image_analysis.get("analysis"):
            final_explanation += f"\n\nImage Insight: {image_analysis['analysis']}"

        # 7. FINAL RESPONSE
        return {
            "prediction": prediction,
            "confidence": confidence,
            "evidence": evidence_list,
            "explanation": final_explanation,
            "image_analysis": image_analysis,
        }
