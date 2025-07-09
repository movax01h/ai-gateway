from contextlib import contextmanager

from fastapi import HTTPException, status
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.integrations.amazon_q.client import AmazonQClientFactory
from ai_gateway.integrations.amazon_q.errors import AWSException
from lib.internal_events.client import InternalEventsClient


@contextmanager
def authorized_q_client(
    current_user: StarletteUser,
    internal_event_client: InternalEventsClient,
    amazon_q_client_factory: AmazonQClientFactory,
    role_arn: str,
    internal_event_category: str,
    internal_event_prefix: str = "request",
):
    if not current_user.can(GitLabUnitPrimitive.AMAZON_Q_INTEGRATION):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to perform action",
        )

    internal_event_client.track_event(
        f"{internal_event_prefix}_{GitLabUnitPrimitive.AMAZON_Q_INTEGRATION}",
        category=internal_event_category,
    )

    try:
        yield amazon_q_client_factory.get_client(
            current_user=current_user,
            role_arn=role_arn,
        )
    except AWSException as e:
        raise e.to_http_exception()
