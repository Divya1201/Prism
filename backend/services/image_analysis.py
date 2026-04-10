import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


class ImageAnalysisService:
    def analyze_image_url(self, image_url: str) -> dict:

        try:
            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": image_url},
                timeout=8
            )

            result = response.json()

            # SUCCESS CASE
            if isinstance(result, list) and len(result) > 0:

                # Take top 3 labels instead of 1
                top_labels = [
                    item.get("label", "unknown")
                    for item in result[:3]
                ]

                top_score = result[0].get("score", 0)

                return {
                    "enabled": True,
                    "labels": top_labels,
                    "confidence": round(top_score, 2),
                    "analysis": f"Image likely contains: {', '.join(top_labels)}"
                }

            # Unexpected format
            return {
                "enabled": True,
                "labels": [],
                "confidence": 0.0,
                "analysis": "Could not interpret image"
            }

        except Exception as e:
            return {
                "enabled": False,
                "error": str(e),
            }
