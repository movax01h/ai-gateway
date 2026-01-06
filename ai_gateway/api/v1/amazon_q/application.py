from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import JSONResponse
from gitlab_cloud_connector import GitLabFeatureCategory

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from ai_gateway.api.v1.amazon_q.typing import (
    ApplicationDeleteRequest,
    ApplicationRequest,
    HealthRequest,
)
from ai_gateway.api.v1.amazon_q.utils import authorized_q_client
from ai_gateway.async_dependency_resolver import (
    get_amazon_q_client_factory,
    get_internal_event_client,
)
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from lib.internal_events import InternalEventsClient
from lib.usage_quota import UsageQuotaEvent

__all__ = [
    "router",
]

router = APIRouter()


@router.post("/application")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
@has_sufficient_usage_quota(
    feature_qualified_name="amazon_q_integration",
    event=UsageQuotaEvent.AMAZON_Q_INTEGRATION,
)
async def oauth_create_application(
    request: Request,  # pylint: disable=unused-argument
    application_request: ApplicationRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    amazon_q_client_factory: Annotated[
        AmazonQClientFactory, Depends(get_amazon_q_client_factory)
    ],
):
    with authorized_q_client(
        current_user=current_user,
        internal_event_client=internal_event_client,
        amazon_q_client_factory=amazon_q_client_factory,
        role_arn=application_request.role_arn,
        internal_event_category=__name__,
    ) as q_client:
        q_client.create_or_update_auth_application(application_request)

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/application/delete")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
@has_sufficient_usage_quota(
    feature_qualified_name="amazon_q_integration",
    event=UsageQuotaEvent.AMAZON_Q_INTEGRATION,
)
async def oauth_delete_application(
    request: Request,  # pylint: disable=unused-argument
    application_request: ApplicationDeleteRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    amazon_q_client_factory: Annotated[
        AmazonQClientFactory, Depends(get_amazon_q_client_factory)
    ],
):
    with authorized_q_client(
        current_user=current_user,
        internal_event_client=internal_event_client,
        amazon_q_client_factory=amazon_q_client_factory,
        role_arn=application_request.role_arn,
        internal_event_category=__name__,
    ) as q_client:
        q_client.delete_o_auth_app_connection()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/application/verify")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
@has_sufficient_usage_quota(
    feature_qualified_name="amazon_q_integration",
    event=UsageQuotaEvent.AMAZON_Q_INTEGRATION,
)
async def validate_auth_app(
    request: Request,  # pylint: disable=unused-argument
    health_request: HealthRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    amazon_q_client_factory: Annotated[
        AmazonQClientFactory, Depends(get_amazon_q_client_factory)
    ],
) -> Response:
    with authorized_q_client(
        current_user=current_user,
        internal_event_client=internal_event_client,
        amazon_q_client_factory=amazon_q_client_factory,
        role_arn=health_request.role_arn,
        internal_event_category=__name__,
        internal_event_prefix="validate_auth",
    ) as q_client:
        # Verify OAuth connection
        response_data = q_client.verify_oauth_connection(health_request)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_data["response"],
    )
