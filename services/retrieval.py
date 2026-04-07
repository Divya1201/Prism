"""Evidence retrieval module using sentence-transformers embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievalResult:
    """One retrieved document and its similarity score."""

    text: str
    score: float


class EvidenceRetriever:
    """In-memory document retriever based on cosine similarity."""

    def __init__(self, documents: Iterable[str] | None = None, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.documents = list(documents) if documents is not None else self._default_documents()
        self.model = SentenceTransformer(model_name)
        self._doc_embeddings = self._build_index(self.documents)

    @staticmethod
    def _default_documents() -> list[str]:
        return [
            "The Federal Reserve kept interest rates unchanged this week while signaling a cautious approach to future cuts as inflation remains above target.",
            "A major technology company announced a new AI chip designed to reduce data-center energy use by up to 30 percent in large-scale inference workloads.",
            "Global oil prices rose after supply disruptions in key shipping routes raised concerns about near-term inventory shortages.",
            "A national health agency reported that seasonal flu cases declined for the third consecutive week following expanded vaccination campaigns.",
            "Scientists published a study showing that urban tree canopies can lower summer surface temperatures by several degrees in densely populated neighborhoods.",
            "The local transit authority approved a multi-year plan to modernize rail signaling systems and improve on-time performance across commuter lines.",
        ]

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _build_index(self, documents: list[str]) -> np.ndarray:
        embeddings = self.model.encode(documents, convert_to_numpy=True)
        return self._l2_normalize(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        query_embedding = self._l2_normalize(query_embedding)[0]

        scores = self._doc_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievalResult(text=self.documents[i], score=float(scores[i]))
            for i in top_indices
        ]


retriever = EvidenceRetriever()


def retrieve_evidence(query: str, top_k: int = 3) -> dict[str, list[dict[str, float | str]]]:
    """Return evidence payload for a given query."""
    results = retriever.retrieve(query, top_k=top_k)
    return {
        "evidence": [
            {"text": item.text, "score": round(item.score, 4)}
            for item in results
        ]
    }
