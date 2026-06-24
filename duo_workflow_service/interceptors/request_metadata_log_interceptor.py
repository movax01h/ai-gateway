from collections.abc import Awaitable, Callable
from typing import Optional

import grpc
import structlog

from duo_workflow_service.interceptors import (
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS,
    X_GITLAB_FEATURE_ENABLEMENT_TYPE,
    X_GITLAB_HOST_NAME,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_IS_A_GITLAB_MEMBER,
    X_GITLAB_IS_GITLAB_MEMBER,
    X_GITLAB_LANGUAGE_SERVER_VERSION,
    X_GITLAB_ORGANIZATION_ID,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_ROOT_NAMESPACE_ID,
    X_GITLAB_VERSION_HEADER,
)


class RequestMetadataLogInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that binds GitLab request metadata to structlog context vars.

    Reads values from gRPC request metadata and binds them to structlog context variables so they are automatically
    included in all log entries emitted during that request.
    """

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Optional[grpc.RpcMethodHandler]:
        """Intercept incoming requests to bind log metadata to structlog context vars."""
        metadata = dict(handler_call_details.invocation_metadata)

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            gitlab_instance_id=metadata.get(X_GITLAB_INSTANCE_ID_HEADER),
            gitlab_host_name=metadata.get(X_GITLAB_HOST_NAME),
            gitlab_realm=metadata.get(X_GITLAB_REALM_HEADER),
            gitlab_version=metadata.get(X_GITLAB_VERSION_HEADER),
            gitlab_language_server_version=metadata.get(
                X_GITLAB_LANGUAGE_SERVER_VERSION
            ),
            gitlab_root_namespace_id=metadata.get(X_GITLAB_ROOT_NAMESPACE_ID),
            gitlab_feature_enabled_by_namespace_ids=metadata.get(
                X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS
            ),
            gitlab_feature_enablement_type=metadata.get(
                X_GITLAB_FEATURE_ENABLEMENT_TYPE
            ),
            gitlab_organization_id=metadata.get(X_GITLAB_ORGANIZATION_ID),
            is_gitlab_team_member=metadata.get(X_GITLAB_IS_GITLAB_MEMBER)
            or metadata.get(X_GITLAB_IS_A_GITLAB_MEMBER),
        )

        return await continuation(handler_call_details)
