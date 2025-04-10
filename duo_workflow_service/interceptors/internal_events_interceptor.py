from datetime import datetime, timezone

import grpc

from duo_workflow_service.interceptors import (
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_REALM_HEADER,
)
from duo_workflow_service.interceptors.correlation_id_interceptor import correlation_id
from duo_workflow_service.internal_events import EventContext, current_event_context


class InternalEventsInterceptor(grpc.aio.ServerInterceptor):

    def __init__(self):
        pass

    async def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)

        context = EventContext(
            realm=metadata.get(X_GITLAB_REALM_HEADER),
            instance_id=metadata.get(X_GITLAB_INSTANCE_ID_HEADER),
            host_name=metadata.get(X_GITLAB_HOST_NAME),
            global_user_id=metadata.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
            context_generated_at=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id.get(),
        )

        current_event_context.set(context)

        return await continuation(handler_call_details)
