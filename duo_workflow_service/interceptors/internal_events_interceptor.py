# pylint: disable=direct-environment-variable-reference

import os
from datetime import datetime, timezone
from typing import List, Optional

import grpc

from duo_workflow_service.interceptors import (
    X_GITLAB_DEPLOYMENT_TYPE,
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS,
    X_GITLAB_FEATURE_ENABLEMENT_TYPE,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_IS_A_GITLAB_MEMBER,
    X_GITLAB_IS_GITLAB_MEMBER,
    X_GITLAB_NAMESPACE_ID,
    X_GITLAB_PROJECT_ID,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_ROOT_NAMESPACE_ID,
    X_GITLAB_USER_ID_HEADER,
)
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.correlation_id_interceptor import correlation_id
from lib.context import gitlab_version, language_server_version
from lib.internal_events import EventContext, current_event_context


def convert_feature_enabled_string_to_list(
    enabled_features: Optional[str] = None,
) -> Optional[List[int]]:
    if not enabled_features or enabled_features == "undefined":
        return None

    feature_list = [int(feature.strip()) for feature in enabled_features.split(",")]
    return list(dict.fromkeys(feature_list))


class InternalEventsInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(
        self, continuation, handler_call_details: grpc.HandlerCallDetails
    ) -> None:
        metadata = dict(handler_call_details.invocation_metadata)

        # LSP and Gitlab monolith are sending different headers https://gitlab.com/gitlab-org/gitlab/-/issues/580618
        is_gitlab_member = metadata.get(X_GITLAB_IS_A_GITLAB_MEMBER, None)
        is_gitlab_member = is_gitlab_member or metadata.get(
            X_GITLAB_IS_GITLAB_MEMBER, None
        )
        is_gitlab_member = (
            is_gitlab_member.lower() == "true" if is_gitlab_member else None
        )

        feature_enabled_by_namespace_ids = metadata.get(
            X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS, None
        )
        feature_enabled_by_namespace_ids = (
            str(feature_enabled_by_namespace_ids)
            if feature_enabled_by_namespace_ids
            else None
        )

        project_id = metadata.get(X_GITLAB_PROJECT_ID)
        project_id = int(project_id) if project_id else None

        namespace_id = metadata.get(X_GITLAB_NAMESPACE_ID)
        namespace_id = int(namespace_id) if namespace_id else None

        # Get language server version from context
        lsp_version = language_server_version.get()
        extra = {}
        if lsp_version and hasattr(lsp_version, "version"):
            extra["lsp_version"] = str(lsp_version.version)

        # Get GitLab instance version from context
        instance_version_value = gitlab_version.get()

        unique_instance_id = None

        user = current_user.get()
        if hasattr(user, "claims") and user.claims:
            unique_instance_id = getattr(user.claims, "gitlab_instance_uid", None)

        context = EventContext(
            realm=metadata.get(X_GITLAB_REALM_HEADER),
            environment=os.environ.get(
                "DUO_WORKFLOW_SERVICE_ENVIRONMENT", "development"
            ),
            source="duo-workflow-service-python",
            instance_id=metadata.get(X_GITLAB_INSTANCE_ID_HEADER),
            unique_instance_id=unique_instance_id,
            host_name=metadata.get(X_GITLAB_HOST_NAME),
            instance_version=instance_version_value,
            global_user_id=metadata.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
            user_id=metadata.get(X_GITLAB_USER_ID_HEADER),
            context_generated_at=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id.get(),
            project_id=project_id,
            feature_enabled_by_namespace_ids=convert_feature_enabled_string_to_list(
                enabled_features=feature_enabled_by_namespace_ids
            ),
            feature_enablement_type=metadata.get(X_GITLAB_FEATURE_ENABLEMENT_TYPE),
            namespace_id=namespace_id,
            ultimate_parent_namespace_id=metadata.get(X_GITLAB_ROOT_NAMESPACE_ID, None)
            or None,
            is_gitlab_team_member=is_gitlab_member,
            deployment_type=metadata.get(X_GITLAB_DEPLOYMENT_TYPE),
            extra=extra,
        )

        current_event_context.set(context)

        return await continuation(handler_call_details)
