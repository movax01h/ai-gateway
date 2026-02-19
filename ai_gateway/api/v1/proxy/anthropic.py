from typing import Annotated

import fastapi
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Request

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE, feature_categories
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from ai_gateway.api.v1.proxy.request import (
    EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    authorize_with_unit_primitive_header,
    verify_project_namespace_metadata,
)
from ai_gateway.async_dependency_resolver import (
    get_abuse_detector,
    get_anthropic_proxy_model_factory,
    get_billing_event_client,
    get_internal_event_client,
    get_proxy_client,
)
from ai_gateway.models.base import KindModelProvider
from ai_gateway.proxy.clients import AnthropicProxyModelFactory, ProxyClient
from lib.billing_events.client import BillingEventsClient
from lib.events import FeatureQualifiedNameStatic
from lib.internal_events import InternalEventsClient
from lib.usage_quota import UsageQuotaEvent

__all__ = [
    "router",
]

log = structlog.stdlib.get_logger("proxy")

router = APIRouter()


@router.post(f"/{KindModelProvider.ANTHROPIC.value}" + "/{path:path}")
@authorize_with_unit_primitive_header()
@verify_project_namespace_metadata()
@feature_categories(EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS)
async def anthropic(
    request: Request,
    background_tasks: BackgroundTasks,  # pylint: disable=unused-argument
    abuse_detector: Annotated[  # pylint: disable=unused-argument
        AbuseDetector, Depends(get_abuse_detector)
    ],
    proxy_client: Annotated[ProxyClient, Depends(get_proxy_client)],
    anthropic_proxy_model_factory: Annotated[
        AnthropicProxyModelFactory, Depends(get_anthropic_proxy_model_factory)
    ],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    billing_event_client: Annotated[  # pylint: disable=unused-argument
        BillingEventsClient, Depends(get_billing_event_client)
    ],
):
    model = await anthropic_proxy_model_factory.factory(request)

    @has_sufficient_usage_quota(
        feature_qualified_name=FeatureQualifiedNameStatic.AIGW_PROXY_USE,
        event=UsageQuotaEvent.AIGW_PROXY_USE,
        model_name=model.model_name,
    )
    async def _do_request(request: fastapi.Request):
        unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]
        internal_event_client.track_event(
            f"request_{unit_primitive}",
            category=__name__,
        )

        return await proxy_client.proxy(request, model)

    return await _do_request(request)
