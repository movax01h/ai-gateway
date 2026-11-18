"""Deprecated import location, moved to ``ai.apps.aigw.app``."""

import warnings

from ai.apps.aigw.app import get_app, get_config

warnings.warn(
    "ai_gateway.app has moved to ai.apps.aigw.app",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["get_app", "get_config"]
