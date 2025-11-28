from typing import override
from urllib.parse import urljoin

import httpx
from aiocache import SimpleMemoryCache, cached
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ai_gateway.api.middleware.base import _PathResolver
from ai_gateway.instrumentators.usage_quota import (
    USAGE_QUOTA_CHECK_TOTAL,
    USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS,
    USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL,
)
from lib.billing_events.context import UsageQuotaEventContext
from lib.feature_flags import FeatureFlag, is_feature_enabled
from lib.internal_events.context import current_event_context


class UsageQuotaMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        customersdot_url: str,
        skip_endpoints: list[str],
        enabled: bool,
        environment: str,
        # The API call to CustomersDot must be completed in under 1 sec
        # to avoid increasing latency for any AI requests.
        request_timeout: float = 1.0,
    ):
        super().__init__(app)

        self.customersdot_url: str = customersdot_url
        self.enabled: bool = enabled
        self.environment: str = environment
        self.path_resolver = _PathResolver.from_optional_list(skip_endpoints)
        self.request_timeout: float = request_timeout

    def is_active(self) -> bool:
        return self.enabled and is_feature_enabled(FeatureFlag.USAGE_QUOTA_LEFT_CHECK)

    @override
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if not self.is_active():
            return await call_next(request)

        if self.path_resolver.skip_path(request.url.path):
            return await call_next(request)

        event_context = current_event_context.get()
        realm = getattr(event_context, "realm", "unknown")
        usage_quota_event_context = UsageQuotaEventContext.from_internal_event(
            event_context
        )

        try:
            has_usage_quota = await self.has_usage_quota_left(usage_quota_event_context)
        except httpx.HTTPError:
            has_usage_quota = True
            USAGE_QUOTA_CHECK_TOTAL.labels(result="fail_open", realm=realm).inc()
            return await call_next(request)

        if not has_usage_quota:
            USAGE_QUOTA_CHECK_TOTAL.labels(result="deny", realm=realm).inc()
            return JSONResponse(
                status_code=402,
                content={
                    "error": "insufficient_credits",
                    "error_code": "USAGE_QUOTA_EXCEEDED",
                    "message": "Consumer does not have sufficient credits for this request",
                },
            )

        response = await call_next(request)
        USAGE_QUOTA_CHECK_TOTAL.labels(result="allow", realm=realm).inc()

        return response

    @cached(ttl=3600, cache=SimpleMemoryCache)
    async def has_usage_quota_left(self, context: UsageQuotaEventContext) -> bool:
        realm = getattr(context, "realm", "unknown")
        params = context.model_dump(exclude_none=True, exclude_unset=True)

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            ) as client:
                url = urljoin(self.customersdot_url, "api/v1/consumers/resolve")
                with USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS.labels(
                    realm=realm
                ).time():
                    response = await client.head(url, params=params)

                # The Customers Portal responds with two HTTP status codes:
                # - Payment Required (402):
                #     returned when the customer does not have enough credits or when the entitlement check fails.
                # - OK (200):
                #     returned when the customer has sufficient credits and the entitlement check passes.
                # For all other HTTP status codes, we allow the request to proceed,
                # but we currently mark them as fail-open.

                status = response.status_code

                if status == httpx.codes.PAYMENT_REQUIRED:
                    USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                        outcome="denied", status=str(status)
                    ).inc()
                    return False

                response.raise_for_status()

                USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                    outcome="success", status="200"
                ).inc()
                return True

        except httpx.TimeoutException as e:
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="timeout", status="timeout"
            ).inc()
            raise e
        except httpx.HTTPStatusError as e:
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="http_error", status=str(e.response.status_code)
            ).inc()
            raise e
        except httpx.RequestError as e:
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="unexpected", status="client_error"
            ).inc()
            raise e

        return False
