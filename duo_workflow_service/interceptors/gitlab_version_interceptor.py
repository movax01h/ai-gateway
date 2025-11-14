import grpc
from gitlab_cloud_connector.auth import X_GITLAB_VERSION_HEADER

from ai_gateway.instrumentators.model_requests import gitlab_version


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
        version = metadata.get(X_GITLAB_VERSION_HEADER.lower(), None)
        if version:
            gitlab_version.set(version)

        return await continuation(handler_call_details)
