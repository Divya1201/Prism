"""Utility helpers for preparing text before analysis."""

import re


def preprocess_text(text: str) -> str:
    """Normalize user text by trimming spaces and removing noisy characters."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s.,!?'-]", "", cleaned)
    return cleaned
