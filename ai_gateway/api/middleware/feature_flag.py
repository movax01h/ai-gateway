from starlette.middleware.base import Request
from starlette_context import context as starlette_context

from ai_gateway.feature_flags import current_feature_flag_context

from .headers import X_GITLAB_ENABLED_FEATURE_FLAGS, X_GITLAB_REALM_HEADER


class FeatureFlagMiddleware:
    """Middleware for feature flags."""

    def __init__(self, app, disallowed_flags: dict = None):
        self.app = app
        self.disallowed_flags = disallowed_flags

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if X_GITLAB_ENABLED_FEATURE_FLAGS not in request.headers:
            await self.app(scope, receive, send)
            return

        enabled_feature_flags = request.headers.get(
            X_GITLAB_ENABLED_FEATURE_FLAGS, ""
        ).split(",")
        enabled_feature_flags = set(enabled_feature_flags)

        if self.disallowed_flags:
            # Remove feature flags that are not supported in the specific realm.
            gitlab_realm = request.headers.get(X_GITLAB_REALM_HEADER, "")
            disallowed_flags = self.disallowed_flags.get(gitlab_realm, set())
            enabled_feature_flags = enabled_feature_flags.difference(disallowed_flags)

        current_feature_flag_context.set(enabled_feature_flags)
        starlette_context["enabled_feature_flags"] = ",".join(enabled_feature_flags)

        await self.app(scope, receive, send)
