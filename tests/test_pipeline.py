from prism_pipeline.pipeline import AnalysisPipeline


def test_pipeline_success_without_image():
    pipeline = AnalysisPipeline()
    payload = pipeline.run("I need a refund because of billing error")

    assert set(payload.keys()) == {
        "prediction",
        "confidence",
        "evidence",
        "explanation",
        "image_analysis",
    }
    assert payload["prediction"] == "billing"
    assert payload["image_analysis"]["enabled"] is False


def test_pipeline_validation_error():
    pipeline = AnalysisPipeline()
    try:
        pipeline.run("   ")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
