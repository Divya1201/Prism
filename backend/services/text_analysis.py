from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="models/bert_model",
    tokenizer="models/bert_model"
)

def analyze_text(text: str, image_url=None):
    result = classifier(text[:512])[0]

    return {
        "prediction": result["label"],
        "confidence": float(result["score"]),
    }
