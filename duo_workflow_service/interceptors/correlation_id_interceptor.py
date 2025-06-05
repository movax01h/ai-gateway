import uuid
from contextvars import ContextVar

import grpc

# Context variable to store correlation ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="undefined")
gitlab_global_user_id: ContextVar[str] = ContextVar(
    "gitlab_global_user_id", default="undefined"
)


class CorrelationIdInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles correlation ID injection and propagation."""

    CORRELATION_ID_KEY = "x-request-id"
    X_GITLAB_GLOBAL_USER_ID_HEADER = "x-gitlab-global-user-id"

    def __init__(self):
        pass

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject correlation ID."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract correlation ID from metadata or generate new one
        request_id = metadata.get(self.CORRELATION_ID_KEY, str(uuid.uuid4()))

        # Set correlation ID in context
        correlation_id.set(request_id)
        gitlab_global_user_id.set(
            metadata.get(self.X_GITLAB_GLOBAL_USER_ID_HEADER, "undefined")
        )

        return await continuation(handler_call_details)
