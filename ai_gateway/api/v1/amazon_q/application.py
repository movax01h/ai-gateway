from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.amazon_q.typing import (
    ApplicationDeleteRequest,
    ApplicationRequest,
    HealthRequest,
)
from ai_gateway.async_dependency_resolver import (
    get_amazon_q_client_factory,
    get_internal_event_client,
)
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.integrations.amazon_q.errors import AWSException
from ai_gateway.internal_events import InternalEventsClient

__all__ = [
    "router",
]

router = APIRouter()


@router.post("/application")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
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
    if not current_user.can(GitLabUnitPrimitive.AMAZON_Q_INTEGRATION):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to perform action",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.AMAZON_Q_INTEGRATION}",
        category=__name__,
    )

    try:
        q_client = amazon_q_client_factory.get_client(
            current_user=current_user,
            role_arn=application_request.role_arn,
        )

        q_client.create_or_update_auth_application(application_request)
    except AWSException as e:
        raise e.to_http_exception()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/application/delete")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
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
    if not current_user.can(GitLabUnitPrimitive.AMAZON_Q_INTEGRATION):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to perform action",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.AMAZON_Q_INTEGRATION}",
        category=__name__,
    )

    try:
        q_client = amazon_q_client_factory.get_client(
            current_user=current_user,
            role_arn=application_request.role_arn,
        )

        q_client.delete_o_auth_app_connection()
    except AWSException as e:
        raise e.to_http_exception()

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/application/verify")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
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
    if not current_user.can(GitLabUnitPrimitive.AMAZON_Q_INTEGRATION):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to perform action",
        )

    internal_event_client.track_event(
        f"validate_auth_{GitLabUnitPrimitive.AMAZON_Q_INTEGRATION}",
        category=__name__,
    )

    try:
        # Get Q client without role_arn as this is just a validation
        q_client = amazon_q_client_factory.get_client(
            current_user=current_user,
            role_arn=health_request.role_arn,
        )

        # Verify OAuth connection
        response_data = q_client.verify_oauth_connection(health_request)
    except AWSException as e:
        raise e.to_http_exception()

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_data["response"],
    )
