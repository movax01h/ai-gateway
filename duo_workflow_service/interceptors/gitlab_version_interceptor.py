from contextvars import ContextVar
from typing import Optional

import grpc

from duo_workflow_service.interceptors import X_GITLAB_VERSION

# Context variable to store GitLab version
gitlab_version: ContextVar[Optional[str]] = ContextVar("gitlab_version", default=None)


class GitLabVersionInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles GitLab version propagation."""

    def __init__(self):
        pass

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to track GitLab version."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract GitLab version from metadata
        version = metadata.get(X_GITLAB_VERSION, None)
        if version:
            gitlab_version.set(version)

        return await continuation(handler_call_details)
