# prism-ml

Reorganized project layout for Prism while preserving existing functionality.

## Structure

- `backend/main.py`: FastAPI entrypoint
- `backend/pipeline.py`: Core orchestration logic
- `backend/services/`: text, retrieval, explanation, and image modules
- `backend/models/request_models.py`: API schemas
- `extension/`: browser extension assets
- `scripts/train_text_classifier.py`: training utility
- `tests/test_pipeline.py`: pipeline sanity test

## Run backend

```bash
cd prism-ml
uvicorn backend.main:app --reload
```
