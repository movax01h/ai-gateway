from contextvars import ContextVar
from typing import Optional

import grpc.aio

X_GITLAB_CLIENT_TYPE_HEADER = "X-Gitlab-Client-Type"

# Context variable to store client type
client_type: ContextVar[Optional[str]] = ContextVar("client_type", default=None)


class ClientTypeInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self):
        pass

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to track client type."""
        metadata = dict(handler_call_details.invocation_metadata)

        result = metadata.get(X_GITLAB_CLIENT_TYPE_HEADER.lower(), None)
        if result:
            client_type.set(result)

        return await continuation(handler_call_details)
