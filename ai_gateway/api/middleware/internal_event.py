from datetime import datetime

from asgi_correlation_id.context import correlation_id
from starlette.datastructures import CommaSeparatedStrings
from starlette.middleware.base import Request
from starlette_context import context as starlette_context

from ai_gateway.api.middleware.base import _PathResolver
from ai_gateway.api.middleware.headers import (
    X_GITLAB_CLIENT_NAME,
    X_GITLAB_CLIENT_TYPE,
    X_GITLAB_CLIENT_VERSION,
    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER,
    X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER,
    X_GITLAB_GLOBAL_USER_ID_HEADER,
    X_GITLAB_HOST_NAME_HEADER,
    X_GITLAB_INSTANCE_ID_HEADER,
    X_GITLAB_INTERFACE,
    X_GITLAB_REALM_HEADER,
    X_GITLAB_ROOT_NAMESPACE_ID,
    X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER,
    X_GITLAB_TEAM_MEMBER_HEADER,
    X_GITLAB_VERSION_HEADER,
)
from ai_gateway.api.middleware_utils import get_valid_namespace_ids
from lib.internal_events import (
    EventContext,
    current_event_context,
    tracked_internal_events,
)


class InternalEventMiddleware:
    def __init__(self, app, skip_endpoints, enabled, environment):
        self.app = app
        self.enabled = enabled
        self.environment = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return

        request = Request(scope)

        if self.path_resolver.skip_path(request.url.path):
            await self.app(scope, receive, send)
            return

        # Get validated namespace IDs that enable feature access for the user
        # This is relevant for requests from gitlab.com, not for self-managed/dedicated instances
        feature_enabled_by_namespace_ids = self._extract_namespace_ids(request)

        # EventContext uses Pydantic which coerces int and string to boolean type
        # Reference: https://docs.pydantic.dev/latest/api/standard_library_types/#booleans
        context = EventContext(
            environment=self.environment,
            source="ai-gateway-python",
            realm=request.headers.get(X_GITLAB_REALM_HEADER),
            instance_id=request.headers.get(X_GITLAB_INSTANCE_ID_HEADER),
            host_name=request.headers.get(X_GITLAB_HOST_NAME_HEADER),
            instance_version=request.headers.get(X_GITLAB_VERSION_HEADER),
            global_user_id=request.headers.get(X_GITLAB_GLOBAL_USER_ID_HEADER),
            is_gitlab_team_member=request.headers.get(X_GITLAB_TEAM_MEMBER_HEADER),  # type: ignore[arg-type]
            client_type=request.headers.get(X_GITLAB_CLIENT_TYPE),
            client_name=request.headers.get(X_GITLAB_CLIENT_NAME),
            client_version=request.headers.get(X_GITLAB_CLIENT_VERSION),
            interface=request.headers.get(X_GITLAB_INTERFACE),
            feature_enabled_by_namespace_ids=feature_enabled_by_namespace_ids,
            feature_enablement_type=request.headers.get(
                X_GITLAB_FEATURE_ENABLEMENT_TYPE_HEADER
            ),
            context_generated_at=datetime.now().isoformat(),
            correlation_id=correlation_id.get(),
            ultimate_parent_namespace_id=request.headers.get(  # type: ignore[arg-type]
                X_GITLAB_ROOT_NAMESPACE_ID, None
            )
            or None,
        )
        current_event_context.set(context)
        tracked_internal_events.set(set())

        await self.app(scope, receive, send)

        starlette_context["tracked_internal_events"] = list(
            tracked_internal_events.get()
        )

    def _extract_namespace_ids(self, request: Request) -> list[int] | None:
        """Extract and validate namespace IDs from request headers.

        Args:
            request: The HTTP request object

        Returns:
            A list of valid namespace IDs or None if an error occurred
        """
        # Try the primary header first
        raw_namespace_ids = list(
            CommaSeparatedStrings(
                request.headers.get(
                    X_GITLAB_FEATURE_ENABLED_BY_NAMESPACE_IDS_HEADER, ""
                )
            )
        )

        # Fall back to legacy header if primary is empty
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/561
        if not raw_namespace_ids:
            raw_namespace_ids = list(
                CommaSeparatedStrings(
                    request.headers.get(X_GITLAB_SAAS_DUO_PRO_NAMESPACE_IDS_HEADER, "")
                )
            )

        try:
            return get_valid_namespace_ids(raw_namespace_ids)
        except Exception:
            return None
