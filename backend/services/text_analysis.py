
"""Service layer for misinformation detection logic."""

from app.utils.preprocessing import preprocess_text


SUSPICIOUS_KEYWORDS = {
    "shocking",
    "secret",
    "cover-up",
    "they don't want you to know",
    "miracle cure",
    "100% guaranteed",
    "hoax",
}


def analyze_text(text: str, image_url: str | None = None) -> dict[str, str | float]:
    """Analyze text with a simple keyword-based baseline classifier."""
    processed_text = preprocess_text(text)

    hits = [keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in processed_text]
    keyword_score = min(len(hits) / 3, 1.0)

    if len(processed_text.split()) < 5:
        keyword_score = min(keyword_score + 0.1, 1.0)

    if image_url:
        keyword_score = max(keyword_score - 0.05, 0.0)

    if keyword_score >= 0.5:
        prediction = "fake"
        confidence = round(0.6 + (keyword_score * 0.35), 2)
        explanation = (
            "The content contains phrases commonly associated with sensational or "
            "misleading claims."
        )
    else:
        prediction = "real"
        confidence = round(0.55 + ((1 - keyword_score) * 0.35), 2)
        explanation = (
            "The content appears neutral and does not contain strong misinformation "
            "indicators in this baseline check."
        )

    if hits:
        explanation += f" Trigger keywords: {', '.join(hits)}."

    return {
        "prediction": prediction,
        "confidence": min(confidence, 0.99),
        "explanation": explanation,
    }
