from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List

import numpy as np
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass(frozen=True)
class KnownImage:
    name: str
    url: str


class ImageAnalysisService:
    """Analyzes image similarity using CLIP embeddings."""

    def __init__(self, similarity_threshold: float = 0.9) -> None:
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._model.eval()
        self._known_images: List[KnownImage] = [
            KnownImage(
                name="city_street",
                url="https://images.unsplash.com/photo-1449824913935-59a10b8d2000",
            ),
            KnownImage(
                name="mountain_lake",
                url="https://images.unsplash.com/photo-1501785888041-af3ef285b470",
            ),
            KnownImage(
                name="office_desk",
                url="https://images.unsplash.com/photo-1498050108023-c5249f4df085",
            ),
        ]
        self._known_embedding_cache: Dict[str, np.ndarray] = {}
        self._similarity_threshold = similarity_threshold

    def _load_image(self, image_url: str) -> Image.Image:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def _get_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)

        embedding = image_features[0].cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        return float(np.dot(emb_a, emb_b))

    def _known_image_embedding(self, known_image: KnownImage) -> np.ndarray:
        if known_image.name in self._known_embedding_cache:
            return self._known_embedding_cache[known_image.name]

        image = self._load_image(known_image.url)
        embedding = self._get_embedding(image)
        self._known_embedding_cache[known_image.name] = embedding
        return embedding

    def analyze_image_url(self, image_url: str) -> Dict[str, float | bool]:
        target_image = self._load_image(image_url)
        target_embedding = self._get_embedding(target_image)

        scores = [
            self._cosine_similarity(target_embedding, self._known_image_embedding(known_image))
            for known_image in self._known_images
        ]
        best_score = max(scores) if scores else 0.0

        return {
            "image_similarity_score": best_score,
            "possible_reuse": best_score >= self._similarity_threshold,
        }
