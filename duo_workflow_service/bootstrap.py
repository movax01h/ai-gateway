"""Bootstrap module for duo_workflow_service.

This module serves as the entry point for the duo-workflow-service command. It imports production-only modules before
starting the server.
"""

# pylint: disable=unused-import
# This is a critical security measure to prevent
# remote arbitrary code execution.
import duo_workflow_service.block_pickle  # noqa: F401
from duo_workflow_service.server import run_app


def run_bootstrap():
    """Bootstrap entry point for duo-workflow-service.

    This function is called when the service starts via the CLI entry point. It ensures block_pickle is imported before
    the server runs.
    """
    run_app()


if __name__ == "__main__":
    run_bootstrap()
