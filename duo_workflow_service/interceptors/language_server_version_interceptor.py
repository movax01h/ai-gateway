import grpc

from ai_gateway.instrumentators.model_requests import language_server_version
from duo_workflow_service.interceptors import X_GITLAB_LANGUAGE_SERVER_VERSION
from lib.language_server import LanguageServerVersion


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
