import uuid
from contextvars import ContextVar

import grpc
from fastapi import WebSocket

from duo_workflow_service.interceptors.websocket_middleware import WebSocketMiddleware

# Context variables to store correlation ID and user ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="undefined")
gitlab_global_user_id: ContextVar[str] = ContextVar(
    "gitlab_global_user_id", default="undefined"
)

CORRELATION_ID_KEY = "x-request-id"
X_GITLAB_GLOBAL_USER_ID_HEADER = "x-gitlab-global-user-id"


def _set_correlation_context(request_id=None, user_id=None):
    if request_id is None:
        request_id = str(uuid.uuid4())

    if user_id is None:
        user_id = "undefined"

    correlation_id.set(request_id)
    gitlab_global_user_id.set(user_id)

    return request_id, user_id


class CorrelationIdInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)

        _set_correlation_context(
            request_id=metadata.get(CORRELATION_ID_KEY),
            user_id=metadata.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
        )

        return await continuation(handler_call_details)


class CorrelationIdMiddleware(WebSocketMiddleware):
    async def __call__(self, websocket: WebSocket):
        _set_correlation_context(
            request_id=websocket.headers.get(CORRELATION_ID_KEY),
            user_id=websocket.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
        )
