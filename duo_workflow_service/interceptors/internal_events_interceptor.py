from datetime import datetime, timezone
from typing import List, Optional

import grpc

from duo_workflow_service.interceptors import (
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_IS_A_GITLAB_MEMBER,
    X_GITLAB_NAMESPACE_ID,
    X_GITLAB_PROJECT_ID,
    X_GITLAB_REALM_HEADER,
)
from duo_workflow_service.interceptors.correlation_id_interceptor import correlation_id
from duo_workflow_service.internal_events import EventContext, current_event_context


def convert_feature_enabled_string_to_list(
    enabled_features: str,
) -> Optional[List[int]]:
    if not enabled_features or enabled_features == "undefined":
        return None

    return [int(feature.strip()) for feature in enabled_features.split(",")]


class InternalEventsInterceptor(grpc.aio.ServerInterceptor):

    def __init__(self):
        pass

    async def intercept_service(self, continuation, handler_call_details):
        metadata = dict(handler_call_details.invocation_metadata)

        is_gitlab_member = metadata.get(X_GITLAB_IS_A_GITLAB_MEMBER, None)
        is_gitlab_member = (
            is_gitlab_member.lower() == "true" if is_gitlab_member else None
        )

        feature_enabled_by_namespace_ids = metadata.get(
            X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS, None
        )

        project_id = metadata.get(X_GITLAB_PROJECT_ID)
        project_id = int(project_id) if project_id else None

        namespace_id = metadata.get(X_GITLAB_NAMESPACE_ID)
        namespace_id = int(namespace_id) if namespace_id else None

        context = EventContext(
            realm=metadata.get(X_GITLAB_REALM_HEADER),
            instance_id=metadata.get(X_GITLAB_INSTANCE_ID_HEADER),
            host_name=metadata.get(X_GITLAB_HOST_NAME),
            global_user_id=metadata.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
            context_generated_at=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id.get(),
            project_id=project_id,
            feature_enabled_by_namespace_ids=convert_feature_enabled_string_to_list(
                enabled_features=feature_enabled_by_namespace_ids
            ),
            namespace_id=namespace_id,
            is_gitlab_team_member=is_gitlab_member,
        )

        current_event_context.set(context)

        return await continuation(handler_call_details)
