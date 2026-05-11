from typing import Optional

from starlette.middleware.base import Request


class _PathResolver:
    def __init__(self, endpoints: list[str]):
        self.endpoints = set(endpoints)

    @classmethod
    def from_optional_list(cls, endpoints: Optional[list] = None) -> "_PathResolver":
        if endpoints is None:
            endpoints = []
        return cls(endpoints)

    def skip_path(self, path: str) -> bool:
        return path in self.endpoints


class InternalEventMiddleware:
    def __init__(self, app, skip_endpoints, enabled, environment):
        self.app = app
        self.enabled = enabled
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

        # Set the distributed tracing LangSmith header to the tracing context, which is sent from
        # Langsmith::RunHelpers of GitLab-Rails/Sidekiq.
        # See https://docs.gitlab.com/ee/development/ai_features/duo_chat.html#tracing-with-langsmith
        # and https://docs.smith.langchain.com/how_to_guides/tracing/distributed_tracing
        await self.app(scope, receive, send)
