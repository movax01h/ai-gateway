import asyncio
from typing import Any, Awaitable, Optional, TypeVar, cast

from fastapi import Request, Response
from prometheus_client import Counter

__all__ = [
    "ClientDisconnectedError",
    "client_disconnected_exception_handler",
    "run_unless_disconnected",
]

T = TypeVar("T")

_DISCONNECT_POLL_INTERVAL_SECONDS = 0.25
_HTTP_499_CLIENT_CLOSED_REQUEST = 499

CLIENT_DISCONNECTS = Counter(
    "client_disconnects_total",
    "Requests abandoned because the client disconnected before the response was ready",
    ["route"],
)


class ClientDisconnectedError(Exception):
    """The client closed the connection while the request was still being processed."""


async def run_unless_disconnected(
    request: Request,
    awaitable: Awaitable[T],
    *,
    enabled: bool = True,
    poll_interval: Optional[float] = None,
) -> T:
    """Await ``awaitable`` in the current task, cancelling it as soon as the request's client disconnects.

    Starlette cancels a streaming response when its client leaves, but the model call that runs before any response
    exists keeps going, including every retry. A side task polls ``request.is_disconnected()`` while that call is in
    flight and cancels this task once the client has gone.

    The call itself stays in the request's task on purpose: a task of its own would get a copy of the context
    variables, and the LLM operations it registers there would be invisible to the code that streams and bills the
    response afterwards.
    """
    if not enabled:
        return await awaitable

    interval = (
        poll_interval
        if poll_interval is not None
        else _DISCONNECT_POLL_INTERVAL_SECONDS
    )
    task = cast("asyncio.Task[Any]", asyncio.current_task())
    cancels_before = task.cancelling()
    disconnected = False

    async def _cancel_on_disconnect() -> None:
        nonlocal disconnected
        while not await request.is_disconnected():
            await asyncio.sleep(interval)
        disconnected = True
        task.cancel()

    watcher = asyncio.create_task(_cancel_on_disconnect())
    try:
        return await awaitable
    except asyncio.CancelledError:
        if not disconnected or task.uncancel() > cancels_before:
            raise
    finally:
        watcher.cancel()

    CLIENT_DISCONNECTS.labels(route=request.url.path).inc()
    raise ClientDisconnectedError()


async def client_disconnected_exception_handler(
    request: Request,  # pylint: disable=unused-argument
    exc: ClientDisconnectedError,  # pylint: disable=unused-argument
) -> Response:
    return Response(status_code=_HTTP_499_CLIENT_CLOSED_REQUEST)
