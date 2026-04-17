from google.cloud import vision


class ImageAnalysisService:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def analyze_image_url(self, image_url: str) -> dict:
        if not image_url:
            return {"enabled": False}

        try:
            image = vision.Image()
            image.source.image_uri = image_url

            #  Web detection (KEY FEATURE)
            web_detection = self.client.web_detection(image=image).web_detection

            #  Labels (context)
            labels = self.client.label_detection(image=image).label_annotations
            label_names = [label.description for label in labels[:3]]

            #  Find matching pages (IMPORTANT)
            matched_pages = []
            if web_detection.pages_with_matching_images:
                for page in web_detection.pages_with_matching_images[:3]:
                    matched_pages.append(page.url)

            #  Interpretation logic
            if matched_pages:
                analysis = (
                    "This image appears across multiple sources online, "
                    "which may indicate reuse or misleading context."
                )
            else:
                analysis = "No strong evidence of reuse found."

            return {
                "enabled": True,
                "labels": label_names,
                "matched_sources": matched_pages,
                "analysis": analysis,
            }

        except Exception as e:
            return {
                "enabled": False,
                "error": str(e),
            }
