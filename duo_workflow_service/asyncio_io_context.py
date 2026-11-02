"""Create asyncio I/O handles outside the request context.

The selector event loop copies the current `contextvars.Context` into each reader and writer `Handle`. Pooled
HTTP sockets are registered from inside the running workflow, so the copy keeps the finished workflow alive
until the connection closes. See docs/performance_testing/performance_profiling.md.

Coroutines and done callbacks are unaffected: Tasks and `add_done_callback` carry their own context. Code that
runs synchronously inside an fd callback (a Task created from `data_received`, `_accept_connection`, the loop
exception handler) now sees the startup context instead. DWS has none of these today; keep it that way, or
register such sockets before `install_clean_io_handle_context` runs.

`add_reader`/`add_writer` take no `context` argument and transports call the private methods, so the wrapper
goes on those two methods of the running loop.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
from asyncio import selector_events
from typing import Any, Callable

import structlog

log = structlog.stdlib.get_logger("asyncio_io_context")

_MARKER = "_dws_clean_io_handle_context_installed"
_REGISTRATION_METHODS = ("_add_reader", "_add_writer")


def _in_context(
    bound: Callable[..., Any], baseline: contextvars.Context
) -> Callable[..., Any]:
    @functools.wraps(bound)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # A fresh copy per call, so nothing that runs inside can change `baseline`.
        return baseline.copy().run(bound, *args, **kwargs)

    return wrapper


def install_clean_io_handle_context(loop: asyncio.AbstractEventLoop) -> bool:
    """Make `loop` create I/O handles in a copy of the context current at install time.

    Returns False, and changes nothing, when `loop` is not a selector loop.
    """
    if getattr(loop, _MARKER, False):
        return True

    if not isinstance(loop, selector_events.BaseSelectorEventLoop) or not all(
        callable(getattr(loop, name, None)) for name in _REGISTRATION_METHODS
    ):
        log.warning(
            "Clean I/O handle context not installed: unsupported event loop",
            loop=type(loop).__qualname__,
        )
        return False

    baseline = contextvars.copy_context()
    for name in _REGISTRATION_METHODS:
        setattr(loop, name, _in_context(getattr(loop, name), baseline))
    setattr(loop, _MARKER, True)

    log.info("Clean I/O handle context installed", loop=type(loop).__qualname__)
    return True
