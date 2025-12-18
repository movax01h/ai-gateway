from typing import override

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ai_gateway.api.middleware.base import _PathResolver
from ai_gateway.instrumentators.usage_quota import USAGE_QUOTA_CHECK_TOTAL
from lib import usage_quota
from lib.billing_events.context import UsageQuotaEventContext
from lib.internal_events.context import current_event_context
from lib.usage_quota.client import UsageQuotaClient


class UsageQuotaMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        customersdot_url: str,
        customersdot_api_user: str | None,
        customersdot_api_token: str | None,
        skip_endpoints: list[str],
        environment: str,
    ):
        super().__init__(app)

        self.usage_quota_client = UsageQuotaClient(
            customersdot_url=customersdot_url,
            customersdot_api_user=customersdot_api_user,
            customersdot_api_token=customersdot_api_token,
        )
        self.environment: str = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)

    @override
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.usage_quota_client.enabled or self.path_resolver.skip_path(
            request.url.path
        ):
            return await call_next(request)

        event_context = current_event_context.get()
        usage_quota_event_context = UsageQuotaEventContext.from_internal_event(
            event_context
        )

        try:
            has_usage_quota = await self.usage_quota_client.check_quota_available(
                usage_quota_event_context
            )
        except usage_quota.UsageQuotaError:
            has_usage_quota = True
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="fail_open", realm=usage_quota_event_context.realm
            ).inc()
            return await call_next(request)

        if not has_usage_quota:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="deny", realm=usage_quota_event_context.realm
            ).inc()
            return JSONResponse(
                status_code=402,
                content={
                    "error": "insufficient_credits",
                    "error_code": "USAGE_QUOTA_EXCEEDED",
                    "message": "Consumer does not have sufficient credits for this request",
                },
            )

        response = await call_next(request)
        USAGE_QUOTA_CHECK_TOTAL.labels(
            result="allow", realm=usage_quota_event_context.realm
        ).inc()

        return response
