"""Service layer for misinformation detection logic."""

from app.utils.preprocessing import preprocess_text


# 8 misinformation categories (aligned with README)
CATEGORY_KEYWORDS = {
    "fabricated": [
        "completely false",
        "made up",
        "fake story",
        "hoax",
        "not real",
    ],
    "false_context": [
        "old image",
        "misleading context",
        "out of context",
        "from years ago",
    ],
    "manipulated": [
        "edited",
        "photoshopped",
        "altered image",
        "deepfake",
    ],
    "imposter": [
        "fake account",
        "pretending to be",
        "impersonating",
    ],
    "false_connection": [
        "clickbait",
        "headline doesn't match",
        "misleading headline",
    ],
    "satire": [
        "satire",
        "parody",
        "not meant to be real",
    ],
    "astroturfing": [
        "bot campaign",
        "coordinated effort",
        "fake engagement",
    ],
    "sponsored": [
        "sponsored",
        "paid promotion",
        "advertisement",
        "ad disguised",
    ],
}


def analyze_text(text: str, image_url: str | None = None) -> dict[str, str | float]:
    """Analyze text and classify into misinformation categories."""

    processed_text = preprocess_text(text)

    # Count keyword matches per category
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in processed_text)
        scores[category] = hits

    # Select best category
    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # If no keywords matched → fallback
    if best_score == 0:
        prediction = "unknown"
        confidence = 0.5
        explanation = (
            "No strong indicators found for a specific misinformation category."
        )
    else:
        prediction = best_category

        # Confidence calculation
        confidence = min(0.5 + best_score * 0.15, 0.95)

        explanation = (
            f"Classified as '{prediction}' based on detected linguistic patterns "
            f"associated with this type of misinformation."
        )

    # Slight adjustment if image present
    if image_url:
        confidence = max(confidence - 0.05, 0.0)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "explanation": explanation,
    }
