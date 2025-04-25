from contextvars import ContextVar
from typing import Optional

import grpc

from duo_workflow_service.interceptors import X_GITLAB_REALM_HEADER

# Context variable to store feature flags
current_feature_flag_context: ContextVar[set] = ContextVar(
    "current_feature_flag_context", default=set()
)


class FeatureFlagInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles feature flags propagation."""

    X_GITLAB_ENABLED_FEATURE_FLAGS = "x-gitlab-enabled-feature-flags"

    def __init__(self, disallowed_flags: Optional[dict] = None):
        self.disallowed_flags = disallowed_flags

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject feature flags context."""
        metadata = dict(handler_call_details.invocation_metadata)

        # Extract enabled feature flags from metadata
        enabled_feature_flags = metadata.get(
            self.X_GITLAB_ENABLED_FEATURE_FLAGS, ""
        ).split(",")
        enabled_feature_flags = set(enabled_feature_flags)

        if self.disallowed_flags:
            # Remove feature flags that are not supported in the specific realm.
            gitlab_realm = metadata.get(X_GITLAB_REALM_HEADER, "")
            disallowed_flags = self.disallowed_flags.get(gitlab_realm, set())
            enabled_feature_flags = enabled_feature_flags.difference(disallowed_flags)

            # Set feature flags in context
        current_feature_flag_context.set(enabled_feature_flags)

        return await continuation(handler_call_details)
