from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from gitlab_cloud_connector import FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.api.v1.proxy.request import authorize_with_unit_primitive_header
from ai_gateway.async_dependency_resolver import (
    get_abuse_detector,
    get_anthropic_proxy_client,
    get_internal_event_client,
)
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import AnthropicProxyClient
from lib.internal_events import InternalEventsClient

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.ANTHROPIC.value}" + "/{path:path}")
@authorize_with_unit_primitive_header()
@feature_categories(FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def anthropic(
    request: Request,
    background_tasks: BackgroundTasks,  # pylint: disable=unused-argument
    # pylint: disable=unused-argument
    abuse_detector: Annotated[AbuseDetector, Depends(get_abuse_detector)],
    anthropic_proxy_client: Annotated[
        AnthropicProxyClient, Depends(get_anthropic_proxy_client)
    ],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
):
    unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]
    internal_event_client.track_event(
        f"request_{unit_primitive}",
        category=__name__,
    )

    return await anthropic_proxy_client.proxy(request)
