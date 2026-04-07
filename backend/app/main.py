"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.routes.analyze import router as analyze_router

app = FastAPI(
    title="Misinformation Detection API",
    version="1.0.0",
    description="Backend service for classifying text as potential misinformation.",
)

app.include_router(analyze_router)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
