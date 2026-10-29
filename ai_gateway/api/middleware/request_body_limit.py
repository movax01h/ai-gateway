import structlog
from fastapi import HTTPException, status
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["RequestBodyLimitMiddleware", "RequestBodyTooLarge"]

log = structlog.stdlib.get_logger("request_body_limit")


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

    A body whose ``Content-Length`` exceeds the limit is rejected before it is read. Otherwise the body is counted as
    it streams in and rejected at the chunk that crosses the limit, before the application buffers it.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, path_prefixes: tuple[str, ...]):
        self.app = app
        self.max_bytes = max_bytes
        self.path_prefixes = path_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(self.path_prefixes):
            await self.app(scope, receive, send)
            return

        declared_length = self._declared_length(scope)
        if declared_length > self.max_bytes:
            self._log_rejection(scope, declared_length)
            await self._reject(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    self._log_rejection(scope, received)
                    raise RequestBodyTooLarge(self.max_bytes)
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    def _declared_length(scope: Scope) -> int:
        return int(Headers(scope=scope).get("content-length", "0"))

    def _log_rejection(self, scope: Scope, body_bytes: int) -> None:
        log.warning(
            "Request body exceeds limit",
            path=scope["path"],
            body_bytes=body_bytes,
            max_bytes=self.max_bytes,
        )

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        exc = RequestBodyTooLarge(self.max_bytes)
        response = JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
        await response(scope, receive, send)
