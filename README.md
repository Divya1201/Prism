# Prism Repository Layout

This repository currently uses a single backend entrypoint and a shared pipeline/services layout.

## Canonical backend

Use the FastAPI app in:

- `backend/main.py`

Run with:

```bash
uvicorn backend.main:app --reload
```

## Key directories

- `backend/`: FastAPI app, request models, service modules, and pipeline integration
- `prism_pipeline/`: lightweight demo pipeline package
- `services/`: shared service implementations
- `tests/`: unit tests for the pipeline package
- `extension/`: browser extension assets
- `scripts/`: utility scripts (for example, training helpers)
