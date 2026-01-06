from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from ai_gateway.api.v1.proxy.request import (
    EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    authorize_with_unit_primitive_header,
    track_billing_event,
    verify_project_namespace_metadata,
)
from ai_gateway.async_dependency_resolver import (
    get_abuse_detector,
    get_anthropic_proxy_client,
    get_billing_event_client,
    get_internal_event_client,
)
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import AnthropicProxyClient
from lib.billing_events.client import BillingEventsClient
from lib.internal_events import InternalEventsClient
from lib.usage_quota import UsageQuotaEvent

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.ANTHROPIC.value}" + "/{path:path}")
@authorize_with_unit_primitive_header()
@track_billing_event
@verify_project_namespace_metadata()
@feature_categories(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
@has_sufficient_usage_quota(
    feature_qualified_name="ai_gateway_proxy_use", event=UsageQuotaEvent.AIGW_PROXY_USE
)
async def anthropic(
    request: Request,
    background_tasks: BackgroundTasks,  # pylint: disable=unused-argument
    abuse_detector: Annotated[  # pylint: disable=unused-argument
        AbuseDetector, Depends(get_abuse_detector)
    ],
    anthropic_proxy_client: Annotated[
        AnthropicProxyClient, Depends(get_anthropic_proxy_client)
    ],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    billing_event_client: Annotated[  # pylint: disable=unused-argument
        BillingEventsClient, Depends(get_billing_event_client)
    ],
):
    unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]
    internal_event_client.track_event(
        f"request_{unit_primitive}",
        category=__name__,
    )

    return await anthropic_proxy_client.proxy(request)
