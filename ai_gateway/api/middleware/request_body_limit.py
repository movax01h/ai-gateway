import time

import structlog
from fastapi import HTTPException, status
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["RequestBodyLimitMiddleware", "RequestBodyTooLarge"]

log = structlog.stdlib.get_logger("request_body_limit")

# Before the 413, the rest of a rejected body is read and dropped, up to this size. A client that sends the whole
# body before it reads the response (Ruby's Net::HTTP, which GitLab Rails uses) otherwise gets a connection reset
# instead of the 413. Cloud Run already stops HTTP/1 request bodies at 32 MiB.
_MAX_DRAIN_BYTES = 64 * 1024 * 1024


class RequestBodyTooLarge(HTTPException):
    """Raised while reading a request body that grows past the configured limit."""

    def __init__(self, max_bytes: int):
        super().__init__(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Request body exceeds {max_bytes} bytes",
            headers={"connection": "close"},
        )


class RequestBodyLimitMiddleware:
    """Answers 413 for request bodies above ``max_bytes`` on paths that start with one of ``path_prefixes``.

    Paths that start with one of ``excluded_prefixes`` are not limited. A body whose ``Content-Length`` exceeds the
    limit is rejected before the application reads it. Otherwise the body is counted as it streams in and rejected at
    the chunk that crosses the limit, before the application buffers it. In both cases the rest of the body is read and
    dropped before the 413, so the client can finish sending and read the response.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_bytes: int,
        path_prefixes: tuple[str, ...],
        excluded_prefixes: tuple[str, ...] = (),
    ):
        self.app = app
        self.max_bytes = max_bytes
        self.path_prefixes = path_prefixes
        self.excluded_prefixes = excluded_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._limits(scope["path"]):
            await self.app(scope, receive, send)
            return

        declared_length = self._declared_length(scope)
        if declared_length > self.max_bytes:
            self._log_rejection(scope, declared_length, declared=True)
            await self._drain(scope, receive)
            await self._reject(
                scope, receive, send, RequestBodyTooLarge(self.max_bytes)
            )
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    self._log_rejection(scope, received, declared=False)
                    # Read the rest here, before the raise: FastAPI answers the HTTPException itself when a route
                    # reads the body, so the 413 can go out before this middleware sees the exception. The chunk
                    # that crossed the limit can be the last one, and then there is nothing more to read.
                    if message.get("more_body", False):
                        await self._drain(scope, receive)
                    raise RequestBodyTooLarge(self.max_bytes)
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge as exc:
            # The exception can come from an outer RequestBodyLimitMiddleware with a lower limit (the code suggestion
            # limit). Answer with the limit in the exception, not the limit of this middleware.
            await self._reject(scope, receive, send, exc)

    def _limits(self, path: str) -> bool:
        return path.startswith(self.path_prefixes) and not path.startswith(
            self.excluded_prefixes
        )

    @staticmethod
    def _declared_length(scope: Scope) -> int:
        return int(Headers(scope=scope).get("content-length", "0"))

    def _log_rejection(self, scope: Scope, body_bytes: int, declared: bool) -> None:
        """Log a rejected body.

        ``declared`` is True when ``body_bytes`` is the ``Content-Length``. It is False when ``body_bytes`` is the
        count at the chunk that crossed the limit.
        """
        log.warning(
            "Request body exceeds limit",
            path=scope["path"],
            body_bytes=body_bytes,
            declared=declared,
            max_bytes=self.max_bytes,
        )

    @staticmethod
    async def _drain(scope: Scope, receive: Receive) -> None:
        """Read the rest of a rejected body and drop it, until the body ends or the client disconnects.

        It stops after ``_MAX_DRAIN_BYTES``, so a client cannot keep the gateway reading forever. It logs the result
        after the "Request body exceeds limit" warning, so a drain that does not end still leaves that warning.
        """
        started = time.perf_counter()
        drained = 0
        outcome = "drain_limit"
        while drained <= _MAX_DRAIN_BYTES:
            message = await receive()
            if message["type"] != "http.request":
                outcome = "client_disconnected"
                break
            drained += len(message.get("body", b""))
            if not message.get("more_body", False):
                outcome = "body_complete"
                break

        log.info(
            "Rejected request body drained",
            path=scope["path"],
            drained_bytes=drained,
            outcome=outcome,
            duration_s=time.perf_counter() - started,
        )

    @staticmethod
    async def _reject(
        scope: Scope, receive: Receive, send: Send, exc: RequestBodyTooLarge
    ) -> None:
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
        await response(scope, receive, send)
