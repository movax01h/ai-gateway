from enum import StrEnum

import structlog

from ai_gateway.instrumentators.usage_quota import USAGE_QUOTA_CHECK_TOTAL
from lib.billing_events.context import UsageQuotaEventContext
from lib.events.base import GLReportingEventContext
from lib.internal_events import current_event_context
from lib.usage_quota.client import UsageQuotaClient
from lib.usage_quota.errors import UsageQuotaError

__all__ = [
    "UsageQuotaEvent",
    "InsufficientCredits",
    "UsageQuotaService",
]


log = structlog.stdlib.get_logger("usage_quota")


class UsageQuotaEvent(StrEnum):
    DAP_FLOW_ON_EXECUTE = "duo_agent_platform_workflow_on_execute"
    DAP_FLOW_ON_GENERATE_TOKEN = "duo_agent_platform_workflow_on_generate_token"
    CODE_SUGGESTIONS_CODE_GENERATIONS = "code_generations"
    CODE_SUGGESTIONS_CODE_COMPLETIONS = "code_completions"
    AMAZON_Q_INTEGRATION = "amazon_q_integration"
    AIGW_PROXY_USE = "ai_gateway_proxy_use"


class InsufficientCredits(Exception):
    def __init__(
        self, message="Consumer does not have sufficient credits for this request."
    ):
        super().__init__(message)


class UsageQuotaService:
    def __init__(
        self,
        customersdot_url: str,
        customersdot_api_user: str | None,
        customersdot_api_token: str | None,
    ):
        self.usage_quota_client = UsageQuotaClient(
            customersdot_url=customersdot_url,
            customersdot_api_user=customersdot_api_user,
            customersdot_api_token=customersdot_api_token,
        )

    async def execute(
        self, gl_context: GLReportingEventContext, event: UsageQuotaEvent
    ):
        if not self.usage_quota_client.enabled:
            log.warning(
                "Usage quota client is disabled",
                correlation_id=getattr(
                    current_event_context.get(), "correlation_id", None
                ),
            )
            return

        event_context = current_event_context.get()
        usage_quota_event_context = UsageQuotaEventContext.from_internal_event(
            event_context
        )

        # Extend usage_quota_event_context with GLReportingEventContext attributes
        extended_context = usage_quota_event_context.model_copy(
            update={
                "event_type": event.value,
                "feature_qualified_name": gl_context.feature_qualified_name,
                "feature_ai_catalog_item": gl_context.feature_ai_catalog_item,
            }
        )

        log.info(
            "Checking usage quota",
            event_type=event.value,
            feature_qualified_name=gl_context.feature_qualified_name,
            correlation_id=getattr(event_context, "correlation_id", None),
        )

        try:
            is_quota_available = await self.usage_quota_client.check_quota_available(
                extended_context
            )
        except UsageQuotaError as e:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="fail_open", realm=usage_quota_event_context.realm
            ).inc()

            log.warning(
                "Usage quota check failed, failing open to allow request",
                realm=usage_quota_event_context.realm,
                error_message=e.message,
                error_type=type(e).__name__,
                correlation_id=getattr(event_context, "correlation_id", None),
            )

            return

        if not is_quota_available:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="deny", realm=usage_quota_event_context.realm
            ).inc()

            raise InsufficientCredits()

        USAGE_QUOTA_CHECK_TOTAL.labels(
            result="allow", realm=usage_quota_event_context.realm
        ).inc()

    async def aclose(self):
        """Cleanup underlying client resources."""
        log.debug("Closing usage quota service")
        await self.usage_quota_client.aclose()
