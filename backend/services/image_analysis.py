import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


class ImageAnalysisService:
    def analyze_image_url(self, image_url: str) -> dict:

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": image_url},
                timeout=10
            )

            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                top_result = result[0]
                label = top_result.get("label", "unknown")
                score = top_result.get("score", 0)

                return {
                    "enabled": True,
                    "label": label,
                    "confidence": round(score, 2),
                }

            return {
                "enabled": True,
                "label": "unknown",
                "confidence": 0.0,
            }

        except Exception:
            return {
                "enabled": False,
                "error": "Image analysis failed",
            }
