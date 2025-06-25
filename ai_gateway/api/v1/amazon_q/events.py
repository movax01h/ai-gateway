from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status
from gitlab_cloud_connector import GitLabFeatureCategory

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.amazon_q.typing import EventRequest
from ai_gateway.api.v1.amazon_q.utils import authorized_q_client
from ai_gateway.async_dependency_resolver import (
    get_amazon_q_client_factory,
    get_internal_event_client,
)
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from lib.internal_events import InternalEventsClient

__all__ = [
    "router",
]

router = APIRouter()


@router.post("/events")
@feature_category(GitLabFeatureCategory.DUO_CHAT)
async def events(
    request: Request,  # pylint: disable=unused-argument
    event_request: EventRequest,
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
        role_arn=event_request.role_arn,
        internal_event_category=__name__,
    ) as q_client:
        q_client.send_event(event_request)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
