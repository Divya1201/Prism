"""Service layer for misinformation detection logic."""

from app.utils.preprocessing import preprocess_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "models/bert_model"
#Safe loading
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
except Exception:
    tokenizer = None
    model = None

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
    """Hybrid analysis: BERT + keyword fallback"""

    processed_text = preprocess_text(text)

    # -------------------------
    # 1. BERT PREDICTION
    # -------------------------
    if tokenizer is None or model is None:
        bert_prediction = None
        bert_confidence = 0.0
    else:
        try:
            inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)

            bert_prediction = model.config.id2label[predicted_class.item()]
            bert_confidence = confidence.item()

        except Exception:
            bert_prediction = None
            bert_confidence = 0.0

    # -------------------------
    # 2. KEYWORD FALLBACK
    # -------------------------
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in processed_text)
        scores[category] = hits

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    # -------------------------
    # 3. DECISION LOGIC
    # -------------------------

    # If BERT is confident → trust it
    if bert_confidence >= 0.6:
        prediction = bert_prediction
        confidence = bert_confidence
        explanation = f"Predicted as '{prediction}' using trained BERT model."

    # Else fallback to keyword logic
    else:
        if best_score == 0:
            prediction = "unknown"
            confidence = 0.5
            explanation = "No strong indicators found."
        else:
            prediction = best_category
            confidence = min(0.5 + best_score * 0.15, 0.95)
            explanation = (
                f"Classified as '{prediction}' using keyword-based fallback."
            )

    # -------------------------
    # 4. IMAGE ADJUSTMENT
    # -------------------------
    if image_url:
        confidence = max(confidence - 0.05, 0.0)

    return {
        "prediction": prediction,
        "confidence": round(min(confidence,0.99), 2),
        "explanation": explanation,
    }
