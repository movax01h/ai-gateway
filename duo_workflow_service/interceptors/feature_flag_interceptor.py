from typing import Dict, Optional, Set

import grpc
from fastapi import WebSocket

from duo_workflow_service.interceptors import X_GITLAB_REALM_HEADER
from duo_workflow_service.interceptors.websocket_middleware import WebSocketMiddleware
from lib.feature_flags.context import current_feature_flag_context

X_GITLAB_ENABLED_FEATURE_FLAGS = "x-gitlab-enabled-feature-flags"


def _set_feature_flag_context(
    metadata: Dict[str, str], disallowed_flags: Optional[Dict[str, Set[str]]] = None
):
    # Extract enabled feature flags from metadata
    enabled_feature_flags = set(
        metadata.get(X_GITLAB_ENABLED_FEATURE_FLAGS, "").split(",")
    )

    if disallowed_flags:
        # Remove feature flags that are not supported in the specific realm.
        gitlab_realm = metadata.get(X_GITLAB_REALM_HEADER, "")
        disallowed_flags_for_realm = disallowed_flags.get(gitlab_realm, set())
        enabled_feature_flags = enabled_feature_flags.difference(
            disallowed_flags_for_realm
        )

    # Set feature flags in context
    current_feature_flag_context.set(enabled_feature_flags)


class FeatureFlagInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles feature flags propagation."""

    def __init__(self, disallowed_flags: Optional[Dict[str, Set[str]]] = None):
        self.disallowed_flags = disallowed_flags

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject feature flags context."""
        metadata = dict(handler_call_details.invocation_metadata)

        _set_feature_flag_context(metadata, self.disallowed_flags)

        return await continuation(handler_call_details)


class FeatureFlagMiddleware(WebSocketMiddleware):
    """Middleware that handles feature flags propagation for WebSockets."""

    def __init__(self, disallowed_flags: Optional[Dict[str, Set[str]]] = None):
        self.disallowed_flags = disallowed_flags

    async def __call__(self, websocket: WebSocket):
        """Process WebSocket connection to inject feature flags context."""
        # Extract headers from WebSocket
        headers = dict(websocket.headers)

        _set_feature_flag_context(headers, self.disallowed_flags)
