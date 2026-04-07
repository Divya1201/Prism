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
from fastapi import FastAPI
from pydantic import BaseModel

from services.retrieval import retrieve_evidence

app = FastAPI()


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    response = {
        "input": payload.text,
    }
    response.update(retrieve_evidence(payload.text, top_k=3))
    return response
