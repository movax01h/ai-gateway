"""Deprecated import location, moved to ``ai.apps.dws.main``."""

import warnings

from ai.apps.dws.main import run_bootstrap

warnings.warn(
    "duo_workflow_service.bootstrap has moved to ai.apps.dws.main",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["run_bootstrap"]


if __name__ == "__main__":
    run_bootstrap()
