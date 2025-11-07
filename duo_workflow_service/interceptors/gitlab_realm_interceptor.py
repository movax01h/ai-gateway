from contextvars import ContextVar
from typing import Optional

import grpc
from gitlab_cloud_connector.auth import X_GITLAB_REALM_HEADER

# Context variable to store GitLab realm
gitlab_realm: ContextVar[Optional[str]] = ContextVar("gitlab_realm", default=None)


class GitLabRealmInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles GitLab realm propagation."""

    def __init__(self):
        pass

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to track GitLab realm."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract GitLab realm from metadata
        realm = metadata.get(X_GITLAB_REALM_HEADER.lower(), None)
        if realm:
            gitlab_realm.set(realm)

        return await continuation(handler_call_details)
