from langsmith.run_helpers import tracing_context
from starlette.middleware.base import Request

from .base import _PathResolver


class DistributedTraceMiddleware:
    """Middleware for distributed tracing."""

    def __init__(self, app, skip_endpoints, environment):
        self.app = app
        self.environment = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        with tracing_context(
            parent=request.headers.get("langsmith-trace"),
            enabled=self.environment == "development",
        ):
            await self.app(scope, receive, send)
