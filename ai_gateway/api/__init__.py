# flake8: noqa

import warnings
from typing import Any

from ai_gateway.api import middleware, monitoring


def __getattr__(name: str) -> Any:
    # Resolved lazily: ai.apps.aigw.server imports this package, so an eager re-export is circular.
    if name == "create_fast_api_server":
        from ai.apps.aigw.server import (  # pylint: disable=import-outside-toplevel
            create_fast_api_server,
        )

        warnings.warn(
            "ai_gateway.api.create_fast_api_server has moved to ai.apps.aigw.server",
            DeprecationWarning,
            stacklevel=2,
        )
        return create_fast_api_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
