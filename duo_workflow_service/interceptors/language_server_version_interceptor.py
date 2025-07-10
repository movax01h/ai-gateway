from contextvars import ContextVar
from typing import Optional

import grpc

from ai_gateway.code_suggestions.language_server import LanguageServerVersion
from duo_workflow_service.interceptors import X_GITLAB_LANGUAGE_SERVER_VERSION

# Context variable to store language server version
language_server_version: ContextVar[Optional[LanguageServerVersion]] = ContextVar(
    "language_server_version", default=None
)


class LanguageServerVersionInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles language server version propagation."""

    def __init__(self):
        pass

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to track language server version."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract language server client version from metadata
        version = metadata.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
        if version:
            language_server_version.set(LanguageServerVersion.from_string(version))

        return await continuation(handler_call_details)
