from __future__ import annotations

import re
import string
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PUNCT_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)
DEFAULT_MODEL_PATH = Path("models/text_classifier.joblib")


def preprocess_text(text: str) -> str:
    """Lowercase and remove punctuation from the input text."""
    lowered = text.lower()
    return lowered.translate(PUNCT_TRANSLATION_TABLE)


def tokenize_text(text: str) -> List[str]:
    """Tokenize preprocessed text with a whitespace-based tokenizer."""
    cleaned = preprocess_text(text)
    return [token for token in re.split(r"\s+", cleaned.strip()) if token]


def build_pipeline() -> Pipeline:
    """Create TF-IDF + Logistic Regression model pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    tokenizer=tokenize_text,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    max_features=50_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, solver="liblinear"),
            ),
        ]
    )


class TextAnalysisService:
    """Service layer for loading model and running fake-news inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self.model: Pipeline = self._load_model(self.model_path)
        self.fake_class_index = self._infer_fake_class_index()

    def _load_model(self, model_path: Path) -> Pipeline:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Train the model first using scripts/train_text_classifier.py"
            )
        return joblib.load(model_path)

    def _infer_fake_class_index(self) -> int:
        classes = list(self.model.classes_)
        normalized_classes = [str(label).strip().lower() for label in classes]

        for candidate in ("fake", "1", "true"):
            if candidate in normalized_classes:
                return normalized_classes.index(candidate)

        # Fallback: binary models often treat the higher class index as positive.
        if len(classes) == 2:
            return 1

        raise ValueError("Could not infer fake class index from model classes.")

    def predict(self, text: str) -> Tuple[str, float, float]:
        fake_probability = float(self.model.predict_proba([text])[0][self.fake_class_index])
        prediction = "fake" if fake_probability >= 0.5 else "real"
        confidence = fake_probability if prediction == "fake" else 1.0 - fake_probability
        return prediction, confidence, fake_probability
