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
