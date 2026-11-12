"""Deprecated import location, moved to ``ai.apps.aigw.main``."""

import warnings

from ai.apps.aigw.main import run_app

warnings.warn(
    "ai_gateway.main has moved to ai.apps.aigw.main",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["run_app"]


if __name__ == "__main__":
    run_app()
