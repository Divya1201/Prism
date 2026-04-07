"""Legacy compatibility module.

The canonical backend is implemented in ``backend.app.main``.
"""

from backend.app.main import app

__all__ = ["app"]
