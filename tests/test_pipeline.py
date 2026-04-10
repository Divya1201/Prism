from backend.pipeline import AnalysisPipeline


def test_pipeline_success_without_image():
    pipeline = AnalysisPipeline()
    payload = pipeline.run("This miracle cure works 100% guaranteed")

    # Check response structure
    assert set(payload.keys()) == {
        "prediction",
        "confidence",
        "evidence",
        "explanation",
        "image_analysis",
    }

    # Check prediction is valid
    assert payload["prediction"] in [
        "fabricated",
        "false_context",
        "manipulated",
        "imposter",
        "false_connection",
        "satire",
        "astroturfing",
        "sponsored",
        "unknown",
    ]

    # Confidence should be valid probability
    assert 0.0 <= payload["confidence"] <= 1.0

    # Evidence should be a list
    assert isinstance(payload["evidence"], list)
    assert len(payload["evidence"]) > 0

    # Explanation should be string
    assert isinstance(payload["explanation"], str)
    assert len(payload["explanation"]) > 0

    # Image analysis disabled
    assert payload["image_analysis"]["enabled"] is False


def test_pipeline_validation_error():
    pipeline = AnalysisPipeline()

    try:
        pipeline.run("   ")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
