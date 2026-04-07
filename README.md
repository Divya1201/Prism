# Prism Repository Layout

This repository previously contained multiple independent API entrypoints (`main.py`, `app.py`, and `app/main.py`) that implemented different `/analyze` behavior. That made local runs inconsistent.

## Canonical backend

Use the FastAPI app in:

- `backend/app/main.py`

Run with:

```bash
uvicorn backend.app.main:app --reload
```

## Compatibility shims

The following legacy files now re-export the canonical app to avoid drift:

- `main.py`
- `app.py`
- `app/main.py`

## Other directories

- `prism_pipeline/`: lightweight demo pipeline + unit tests
- `services/`: model/pipeline support services used by legacy and scripts
- `prism-ml/`: packaged variant maintained for compatibility
- `extension/`: browser extension assets
