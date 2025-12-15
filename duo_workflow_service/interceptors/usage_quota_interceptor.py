from typing import Callable, Optional

import grpc
from grpc.aio import ServerInterceptor, ServicerContext

from ai_gateway.instrumentators.usage_quota import USAGE_QUOTA_CHECK_TOTAL
from lib import usage_quota
from lib.billing_events.context import UsageQuotaEventContext
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context
from lib.internal_events.context import current_event_context
from lib.usage_quota.client import UsageQuotaClient


class UsageQuotaInterceptor(ServerInterceptor):
    def __init__(
        self,
        customersdot_url: str,
    ):
        self.usage_quota_client = UsageQuotaClient(customersdot_url=customersdot_url)

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
        usage_quota_event_context = UsageQuotaEventContext.from_internal_event(
            event_context
        )

        try:
            is_quota_available = await self.usage_quota_client.check_quota_available(
                usage_quota_event_context
            )
        except usage_quota.UsageQuotaError:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="fail_open", realm=usage_quota_event_context.realm
            ).inc()
            return await continuation(handler_call_details)

        if not is_quota_available:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="deny", realm=usage_quota_event_context.realm
            ).inc()
            return self._abort_handler(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                "Consumer does not have sufficient credits for this request. "
                "Error code: USAGE_QUOTA_EXCEEDED",
            )

        USAGE_QUOTA_CHECK_TOTAL.labels(
            result="allow", realm=usage_quota_event_context.realm
        ).inc()
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
