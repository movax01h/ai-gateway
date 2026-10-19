"""Bootstrap module for duo_workflow_service.

This module serves as the entry point for the duo-workflow-service command. It imports production-only modules before
starting the server.
"""

# pylint: disable=unused-import
# Importing this early patches Jinja2's default filter table to block
# reversed()-on-dict, a CPython memory-safety bug (python/cpython#154709).
# Must run before anything constructs a jinja2 Environment.
import ai_gateway.prompts.base  # noqa: F401

# This is a critical security measure to prevent
# remote arbitrary code execution.
import duo_workflow_service.block_pickle  # noqa: F401
from duo_workflow_service.server import run_app


def run_bootstrap():
    """Bootstrap entry point for duo-workflow-service.

    This function is called when the service starts via the CLI entry point. It ensures block_pickle and the Jinja2
    filter hardening in ai_gateway.prompts.base are imported before the server runs.
    """
    run_app()


if __name__ == "__main__":
    run_bootstrap()
