import grpc
from gitlab_cloud_connector.auth import X_GITLAB_REALM_HEADER

from ai_gateway.instrumentators.model_requests import gitlab_realm


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
