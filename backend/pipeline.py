"""Core analysis pipeline orchestration."""

from __future__ import annotations

from typing import Any


class AnalysisPipeline:
    """Coordinates baseline text analysis with retrieval/explanation placeholders."""

    _LABEL_KEYWORDS: dict[str, tuple[str, ...]] = {
        "billing": ("bill", "billing", "refund", "charge", "invoice", "payment"),
        "technical": ("error", "bug", "crash", "login", "broken", "issue"),
        "shipping": ("shipping", "delivery", "delayed", "package", "tracking"),
    }

    _EVIDENCE_LIBRARY: dict[str, list[str]] = {
        "billing": [
            "Customer support policy confirms refunds for duplicate or incorrect charges.",
            "Recent tickets with billing keywords are usually resolved by account review.",
            "A billing dispute workflow requires invoice ID and transaction timestamp.",
        ],
        "technical": [
            "Technical incidents are triaged by severity and affected component.",
            "Known fixes for login and crash issues are documented in support runbooks.",
            "Error reports with reproducible steps are prioritized for release patches.",
        ],
        "shipping": [
            "Shipping delays are commonly caused by carrier handoff backlogs.",
            "Tracking mismatches often resolve within one carrier scan cycle.",
            "Expedited replacement can be approved when delivery SLA is exceeded.",
        ],
        "general": [
            "General requests are routed to a support specialist for manual review.",
            "Additional customer context improves classification confidence.",
            "Escalation paths are available when automated triage is inconclusive.",
        ],
    }

    def run(self, text: str, image_url: str | None = None) -> dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text must not be empty")

        lowered = cleaned.lower()
        prediction = "general"
        hits = 0
        for label, keywords in self._LABEL_KEYWORDS.items():
            label_hits = sum(1 for keyword in keywords if keyword in lowered)
            if label_hits > hits:
                prediction = label
                hits = label_hits

        confidence = min(0.55 + (hits * 0.15), 0.95)

        evidence = [
            {"text": item, "score": round(0.9 - (idx * 0.1), 4)}
            for idx, item in enumerate(self._EVIDENCE_LIBRARY.get(prediction, self._EVIDENCE_LIBRARY["general"]))
        ]
        explanation = (
            f"Classified as '{prediction}' based on detected support intent signals "
            f"in the submitted text."
        )

        image_payload = (
            {
                "enabled": True,
                "image_similarity_score": 0.0,
                "possible_reuse": False,
                "source": image_url,
            }
            if image_url
            else {"enabled": False}
        )

        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "evidence": evidence,
            "explanation": explanation,
            "image_analysis": image_payload,
        }
