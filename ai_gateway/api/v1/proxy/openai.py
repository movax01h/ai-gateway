from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.api.v1.proxy.request import (
    EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    authorize_with_unit_primitive_header,
    track_billing_event,
)
from ai_gateway.async_dependency_resolver import (
    get_abuse_detector,
    get_billing_event_client,
    get_internal_event_client,
    get_openai_proxy_client,
)
from ai_gateway.proxy.clients import OpenAIProxyClient
from lib.billing_events.client import BillingEventsClient
from lib.internal_events import InternalEventsClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post("/openai" + "/{path:path}")
@authorize_with_unit_primitive_header()
@track_billing_event
@feature_categories(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def openai(
    request: Request,
    background_tasks: BackgroundTasks,  # pylint: disable=unused-argument
    abuse_detector: Annotated[  # pylint: disable=unused-argument
        AbuseDetector, Depends(get_abuse_detector)
    ],
    openai_proxy_client: Annotated[OpenAIProxyClient, Depends(get_openai_proxy_client)],
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

    return await openai_proxy_client.proxy(request)
