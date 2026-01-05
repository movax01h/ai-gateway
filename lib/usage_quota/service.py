from enum import StrEnum

from ai_gateway.instrumentators.usage_quota import USAGE_QUOTA_CHECK_TOTAL
from lib.billing_events.context import UsageQuotaEventContext
from lib.events.base import GLReportingEventContext
from lib.internal_events import current_event_context
from lib.usage_quota.client import UsageQuotaClient
from lib.usage_quota.errors import UsageQuotaError

__all__ = [
    "EventType",
    "InsufficientCredits",
    "UsageQuotaService",
]


class EventType(StrEnum):
    DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE = "duo_agent_platform_workflow_on_execute"


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

    async def execute(self, _context: GLReportingEventContext, _event: EventType):
        if not self.usage_quota_client.enabled:
            return

        event_context = current_event_context.get()
        usage_quota_event_context = UsageQuotaEventContext.from_internal_event(
            event_context
        )

        try:
            is_quota_available = await self.usage_quota_client.check_quota_available(
                usage_quota_event_context
            )
        except UsageQuotaError:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="fail_open", realm=usage_quota_event_context.realm
            ).inc()
            return

        if not is_quota_available:
            USAGE_QUOTA_CHECK_TOTAL.labels(
                result="deny", realm=usage_quota_event_context.realm
            ).inc()

            raise InsufficientCredits()

        USAGE_QUOTA_CHECK_TOTAL.labels(
            result="allow", realm=usage_quota_event_context.realm
        ).inc()
