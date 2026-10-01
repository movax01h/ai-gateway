import asyncio
from typing import Dict, Optional

import structlog

__all__ = [
    "log_exception",
    "log_workflow_failure",
]

log = structlog.stdlib.get_logger("exceptions")


def log_exception(ex: BaseException, extra: Optional[Dict] = None) -> None:
    """Log the exception with the correlation ID.

    Args:
    ex (``Exception``):
        Raised exception during application runtime.
    extra (``dict``, `optional`):
        Additional metadata for the exception.
    """
    status_code = getattr(ex, "code", None)
    exception_class = type(ex).__name__

    if extra is None:
        extra = {}

    log.error(
        str(ex),
        status_code=status_code,
        exception_class=exception_class,
        additional_details=extra,
        exc_info=ex,
        stack_info=True,
    )


def log_workflow_failure(ex: BaseException, extra: Optional[Dict] = None) -> None:
    """Log the error that ended a workflow run.

    A ``CancelledError`` means the run was cancelled from outside, typically because the client closed the stream
    or the RPC was aborted. Nothing failed inside the workflow, so it is logged at info level. Every other error is
    logged with ``log_exception``.

    Args:
    ex (``BaseException``):
        The error that ended the workflow run.
    extra (``dict``, `optional`):
        Additional metadata for the error.
    """
    if isinstance(ex, asyncio.CancelledError):
        log.info(
            "Workflow run cancelled",
            reason=str(ex),
            exception_class=type(ex).__name__,
            additional_details=extra or {},
        )
        return

    log_exception(ex, extra=extra)
