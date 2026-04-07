"""Legacy compatibility entrypoint.

This repository had multiple competing API entrypoints. To avoid drift,
we now expose the canonical app from ``backend.app.main``.
"""

from backend.app.main import app

__all__ = ["app"]
