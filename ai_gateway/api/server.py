"""Deprecated import location, moved to ``ai.apps.aigw.server``."""

import warnings

from ai.apps.aigw.server import CONTAINER_APPLICATION_MODULES, create_fast_api_server

warnings.warn(
    "ai_gateway.api.server has moved to ai.apps.aigw.server",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CONTAINER_APPLICATION_MODULES", "create_fast_api_server"]
