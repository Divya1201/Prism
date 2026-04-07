"""Legacy compatibility entrypoint.

Use ``backend.app.main:app`` as the canonical ASGI application.
"""

from backend.app.main import app

__all__ = ["app"]
