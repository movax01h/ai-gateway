import asyncio
import contextlib
from typing import Awaitable, Optional, TypeVar

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
    """Await ``awaitable``, cancelling it as soon as the request's client disconnects.

    Starlette cancels a streaming response when its client leaves, but the model call that runs before any response
    exists keeps going, including every retry. Polling ``request.is_disconnected()`` while that call is in flight
    closes the gap for streaming and non-streaming requests alike.
    """
    if not enabled:
        return await awaitable

    interval = (
        poll_interval
        if poll_interval is not None
        else _DISCONNECT_POLL_INTERVAL_SECONDS
    )
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                return task.result()
            if await request.is_disconnected():
                break
    finally:
        if not task.done():
            await _cancel(task)

    CLIENT_DISCONNECTS.labels(route=request.url.path).inc()
    raise ClientDisconnectedError()


async def _cancel(task: "asyncio.Future[T]") -> None:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def client_disconnected_exception_handler(
    request: Request,  # pylint: disable=unused-argument
    exc: ClientDisconnectedError,  # pylint: disable=unused-argument
) -> Response:
    return Response(status_code=_HTTP_499_CLIENT_CLOSED_REQUEST)
