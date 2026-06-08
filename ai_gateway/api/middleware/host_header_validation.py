"""Sanitise malformed `Host` headers before they reach the application.

starlette (< 1.0.1) reconstructs `request.url` from the `Host` header
without validation. A header such as `example.com/x?` makes
`request.url.path` disagree with `scope["path"]`, enabling middleware
bypass (X41-2026-002 / PYSEC-2026-161).

This middleware mirrors the fix from starlette 1.0.1
(https://github.com/Kludex/starlette/pull/3279): when the `Host` header
does not match the RFC-allowed character set the header is stripped from the
scope so that URL construction falls back to `scope["server"]`. The
request is allowed to continue, which matches upstream behaviour and avoids
breaking callers that would otherwise see a 400.

Remove this middleware once the project is upgraded to starlette >= 1.0.1.
"""

import re

from starlette.types import ASGIApp, Receive, Scope, Send

# Allowlist copied from starlette 1.0.1 (Kludex/starlette#3279).
# Accepts: domain/IPv4 labels (alphanumeric, hyphens, dots) and IPv6
# literals (hex digits, colons, dots in square brackets), with optional port.
_HOST_RE = re.compile(
    r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9.:]+\])(?::[0-9]+)?$",
    re.IGNORECASE,
)


class HostHeaderValidationMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = list(scope.get("headers", ()))
        host = next(
            (v.decode("latin-1", errors="replace") for k, v in headers if k == b"host"),
            None,
        )

        if host is not None and not _HOST_RE.fullmatch(host):
            scope = {**scope, "headers": [(k, v) for k, v in headers if k != b"host"]}

        await self.app(scope, receive, send)
