from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# DATA STRUCTURE
@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float

# -------------------------
# WEB SEARCH FUNCTION
# -------------------------
def fetch_web_evidence(query: str, top_k: int = 5) -> List[str]:
    """
    Fetch real-world evidence using DuckDuckGo API
    """
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        results = []

        if "RelatedTopics" in data:
            for item in data["RelatedTopics"]:
                if isinstance(item, dict) and "Text" in item:
                    results.append(item["Text"])

                    if len(results) >= top_k:
                        break

        return results

    except Exception:
        # Fail silently → fallback will handle
        return []

# -----------------------
# RETRIEVER CLASS
# -----------------------
class EvidenceRetriever:
    def __init__(
        self, documents: Iterable[str] | None = None, 
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        # Fallback static docs
        self.documents = list(documents) if documents is not None else self._default_documents()

        # Embedding model
        self.model = SentenceTransformer(model_name)
        # self._doc_embeddings = self._build_index(self.documents)

    @staticmethod
    def _default_documents() -> list[str]:
        return [
            "Fact-checking organizations verify claims using multiple credible sources.",
            "Misinformation often spreads faster on social media than verified information.",
            "Images taken out of context can mislead viewers about real events.",
            "Satirical content is not intended to be factual but may be misinterpreted.",
            "Sponsored content may resemble news but is paid promotion.",
        ]

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return self._l2_normalize(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        if not query.strip():
            return []

        # -------------------------
        # 1. FETCH REAL WEB DATA
        # -------------------------
        web_docs = fetch_web_evidence(query, top_k=5)
        
        # Use web docs if available, else fallback
        documents = web_docs if web_docs else self.documents

        # -------------------------
        # 2. EMBEDDINGS + SIMILARITY
        # -------------------------
        doc_embeddings = self._encode(documents)
        query_embedding = self._encode([query])[0]

        scores = doc_embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievalResult(text=documents[i], score=float(scores[i]))
            for i in top_indices
        ]

# -------------------------
# GLOBAL RETRIEVER INSTANCE
# -------------------------
retriever = EvidenceRetriever()


# -------------------------
# PUBLIC FUNCTION (USED BY PIPELINE)
# -------------------------
def retrieve_evidence(
    query: str, top_k: int = 3
) -> dict[str, list[dict[str, float | str]]]:

    results = retriever.retrieve(query, top_k=top_k)

    return {
        "evidence": [
            {"text": item.text, "score": round(item.score, 4)}
            for item in results
        ]
    }
