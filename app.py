import logging
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from prism_pipeline.pipeline import AnalysisPipeline

# Configure application-wide logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
pipeline = AnalysisPipeline()


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    """Run the complete analysis pipeline and return final JSON."""
    try:
        payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
        text: str = payload.get("text", "")
        image: Optional[Dict[str, Any]] = payload.get("image")

        logger.info("/analyze request received")
        result = pipeline.run(text=text, image=image)
        logger.info("/analyze request completed", extra={"prediction": result["prediction"]})
        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Validation error in /analyze", exc_info=True)
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Unexpected error in /analyze")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
from __future__ import annotations

from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.explainer import ExplainerService


app = FastAPI()
explainer_service = ExplainerService()


class AnalyzeRequest(BaseModel):
    input_text: str = Field(..., description="Claim text to analyze")
    retrieved_evidence: List[str] = Field(default_factory=list)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, str]:
    return explainer_service.generate_explanation(
        input_text=request.input_text,
        retrieved_evidence=request.retrieved_evidence,
    )
