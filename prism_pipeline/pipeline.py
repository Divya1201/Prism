import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    label: str
    confidence: float


class TextPreprocessor:
    """Step 1: Clean and normalize user text for downstream processing."""

    def run(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text.strip())
        normalized = cleaned.lower()
        logger.debug("Preprocessing complete", extra={"cleaned_text": normalized})
        return normalized


class Classifier:
    """Step 2: Lightweight keyword classifier for demo purposes."""

    def run(self, text: str) -> ClassificationResult:
        if any(token in text for token in ["refund", "chargeback", "billing"]):
            return ClassificationResult(label="billing", confidence=0.88)
        if any(token in text for token in ["error", "bug", "crash", "failure"]):
            return ClassificationResult(label="technical_issue", confidence=0.84)
        return ClassificationResult(label="general_inquiry", confidence=0.65)


class Retriever:
    """Step 3: Retrieve evidence snippets relevant to predicted label."""

    _knowledge_base: Dict[str, List[str]] = {
        "billing": [
            "Billing disputes can be resolved within 5-7 business days.",
            "Refund requests require order id and payment reference.",
        ],
        "technical_issue": [
            "Collect app version, stack trace, and repro steps.",
            "Service status should be checked before escalation.",
        ],
        "general_inquiry": [
            "General requests are routed to support queue within 24 hours.",
            "Include account identifier for faster triage.",
        ],
    }

    def run(self, label: str) -> List[str]:
        evidence = self._knowledge_base.get(label, [])
        logger.debug("Retrieval complete", extra={"label": label, "evidence_count": len(evidence)})
        return evidence


class ExplanationGenerator:
    """Step 4: Produce a human-readable rationale from model output and evidence."""

    def run(self, classification: ClassificationResult, evidence: List[str]) -> str:
        joined_evidence = "; ".join(evidence) if evidence else "No evidence available"
        return (
            f"The input was classified as '{classification.label}' "
            f"with confidence {classification.confidence:.2f} based on retrieved context: {joined_evidence}."
        )


class ImageAnalyzer:
    """Step 5: Optional image analysis placeholder."""

    def run(self, image_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not image_payload:
            return {"enabled": False, "summary": "No image provided"}

        mime_type = image_payload.get("mime_type", "unknown")
        size_bytes = len(image_payload.get("data", ""))
        logger.debug("Image analysis complete", extra={"mime_type": mime_type, "size_bytes": size_bytes})

        # Placeholder analysis result. Replace with CV model integration when available.
        return {
            "enabled": True,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "summary": "Image received and validated for downstream CV processing.",
        }


class AnalysisPipeline:
    """Orchestrates all required modules into a complete /analyze pipeline."""

    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.classifier = Classifier()
        self.retriever = Retriever()
        self.explainer = ExplanationGenerator()
        self.image_analyzer = ImageAnalyzer()

    def run(self, text: str, image: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("'text' is required and must be non-empty")

        # 1) Text preprocessing
        normalized_text = self.preprocessor.run(text)

        # 2) Classification
        classification = self.classifier.run(normalized_text)

        # 3) Retrieval
        evidence = self.retriever.run(classification.label)

        # 4) Explanation generation
        explanation = self.explainer.run(classification, evidence)

        # 5) Optional image analysis
        image_analysis = self.image_analyzer.run(image)

        return {
            "prediction": classification.label,
            "confidence": classification.confidence,
            "evidence": evidence,
            "explanation": explanation,
            "image_analysis": image_analysis,
        }
