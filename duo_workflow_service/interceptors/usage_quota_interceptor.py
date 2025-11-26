import os
from typing import Callable, Optional, cast
from urllib.parse import urljoin

import grpc
import httpx
from aiocache import SimpleMemoryCache, cached
from grpc.aio import ServerInterceptor, ServicerContext

from ai_gateway.instrumentators.usage_quota import (
    USAGE_QUOTA_CHECK_TOTAL,
    USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS,
    USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL,
)
from lib.billing_events.context import UsageQuotaEventContext
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context
from lib.internal_events.context import current_event_context

# pylint: disable=direct-environment-variable-reference
CACHE_TTL = (
    5 if os.environ.get("AIGW_MOCK_USAGE_CREDITS", "").lower() == "true" else 3600
)
# pylint: enable=direct-environment-variable-reference


class UsageQuotaInterceptor(ServerInterceptor):
    def __init__(
        self,
        # The API call to CustomersDot must be completed in under 1 sec
        # to avoid increasing latency for any AI requests.
        customersdot_request_timeout: float = 1.0,
    ):
        self.customersdot_request_timeout = customersdot_request_timeout

    async def intercept_service(
        self,
        continuation: Callable,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Intercept gRPC requests to enforce usage quota checks.

        Args:
            continuation: Function to call to continue processing the request
            handler_call_details: Details about the handler being invoked

        Returns:
            The RPC method handler, either the original or an abort handler
        """
        if FeatureFlag.USAGE_QUOTA_LEFT_CHECK not in current_feature_flag_context.get():
            return await continuation(handler_call_details)

        event_context = current_event_context.get()
        realm = getattr(event_context, "realm", "unknown")
        usage_quota_event_context = UsageQuotaEventContext.from_internal_event(
            event_context
        )

        try:
            has_usage_quota = await self.has_usage_quota_left(usage_quota_event_context)
        except httpx.HTTPError:
            USAGE_QUOTA_CHECK_TOTAL.labels(result="fail_open", realm=realm).inc()
            return await continuation(handler_call_details)

        if not has_usage_quota:
            USAGE_QUOTA_CHECK_TOTAL.labels(result="deny", realm=realm).inc()
            return self._abort_handler(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Consumer does not have sufficient credits for this request. "
                "Error code: USAGE_QUOTA_EXCEEDED",
            )

        USAGE_QUOTA_CHECK_TOTAL.labels(result="allow", realm=realm).inc()
        return await continuation(handler_call_details)

    def _abort_handler(
        self,
        code: grpc.StatusCode,
        message: str,
        error_metadata: Optional[dict[str, str]] = None,
    ) -> grpc.RpcMethodHandler:
        """Create a handler that aborts with structured error metadata.

        This sets trailing metadata with structured error information that
        clients can parse to understand the error details.

        Args:
            code: gRPC status code to return
            message: Human-readable error message
            error_metadata: Key-value pairs to include in trailing metadata

        Returns:
            An RPC method handler that aborts the request with metadata
        """

        async def handler(_request: object, context: ServicerContext) -> object:
            if error_metadata:
                context.set_trailing_metadata(list(error_metadata.items()))

            await context.abort(code, message)
            return None

        return grpc.unary_unary_rpc_method_handler(handler)

    @cached(ttl=CACHE_TTL, cache=SimpleMemoryCache)
    async def has_usage_quota_left(self, context: UsageQuotaEventContext) -> bool:
        """Check if the consumer has usage quota left.

        This method is cached with a TTL of 1 hour to reduce load on CustomersDot.

        Args:
            context: Usage quota event context containing consumer information

        Returns:
            True if the consumer has sufficient credits, False otherwise

        Raises:
            httpx.HTTPError: If there's an error communicating with CustomersDot
        """
        realm = getattr(context, "realm", "unknown")
        params = context.model_dump(exclude_none=True, exclude_unset=True)
        # pylint: disable=direct-environment-variable-reference
        customer_portal_url = (
            os.environ.get("AIGW_MOCK_CRED_CD_URL")
            if os.environ.get("AIGW_MOCK_USAGE_CREDITS", "").lower() == "true"
            and os.environ.get("AIGW_MOCK_CRED_CD_URL")
            else os.environ.get(
                "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL",
                "https://customers.gitlab.com",
            )
        )

        # pylint: enable=direct-environment-variable-reference

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.customersdot_request_timeout),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            ) as client:
                url = urljoin(
                    cast(str, customer_portal_url),
                    cast(str, "api/v1/consumers/resolve"),
                )

                with USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS.labels(
                    realm=realm
                ).time():
                    response = await client.head(url, params=params)

                # The Customers Portal responds with following HTTP status codes:
                # - Payment Required (402):
                #     returned when the customer does not have enough credits.
                # - Forbidden (403):
                #     returned when the entitlement check fails.
                # - OK (200):
                #     returned when the customer has sufficient credits
                #     and the entitlement check passes.
                # For all other HTTP status codes, we raise an exception,
                # which causes the request to fail open.

                status = response.status_code

                if status in [httpx.codes.PAYMENT_REQUIRED, httpx.codes.FORBIDDEN]:
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
