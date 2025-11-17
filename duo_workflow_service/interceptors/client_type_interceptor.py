import grpc.aio

from ai_gateway.instrumentators.model_requests import client_type

X_GITLAB_CLIENT_TYPE_HEADER = "X-Gitlab-Client-Type"


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
