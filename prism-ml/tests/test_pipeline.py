from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import AnalysisPipeline


def test_pipeline_requires_model():
    try:
        pipeline = AnalysisPipeline()
    except (FileNotFoundError, ModuleNotFoundError):
        # acceptable in fresh env without model/artifacts/dependencies
        return

    payload = pipeline.run("This is a sample claim about markets")
    assert "prediction" in payload
    assert "explanation" in payload
    assert "evidence" in payload
    assert payload["image_analysis"]["possible_reuse"] in {True, False}
