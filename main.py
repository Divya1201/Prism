from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl

from services.image_analysis import ImageAnalysisService


app = FastAPI()
image_analysis_service = ImageAnalysisService()


class AnalyzeRequest(BaseModel):
    image_url: HttpUrl


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    try:
        return image_analysis_service.analyze_image_url(str(request.image_url))
    except Exception as exc:  # surface remote/load/model errors as bad request
        raise HTTPException(status_code=400, detail=f"Unable to analyze image: {exc}") from exc
