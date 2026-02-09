from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
    TokenAuthority,
    UserClaims,
)

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from ai_gateway.api.v1.code.typing import Token
from ai_gateway.async_dependency_resolver import (
    get_internal_event_client,
    get_token_authority,
)
from lib.context import StarletteUser, get_current_user
from lib.events import FeatureQualifiedNameStatic
from lib.internal_events import InternalEventsClient
from lib.usage_quota import UsageQuotaEvent

__all__ = [
    "router",
]


log = structlog.stdlib.get_logger("user_access_token")

router = APIRouter()


@router.post("/user_access_token")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@has_sufficient_usage_quota(
    feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
    event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
)
async def user_access_token(
    request: Request,  # pylint: disable=unused-argument
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    token_authority: Annotated[TokenAuthority, Depends(get_token_authority)],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
    x_gitlab_project_id: Annotated[
        Optional[str], Header()
    ] = None,  # This is the value of X_GITLAB_PROJECT_ID_HEADER
    x_gitlab_namespace_id: Annotated[
        Optional[str], Header()
    ] = None,  # This is the value of X_GITLAB_NAMESPACE_ID_HEADER
    x_gitlab_root_namespace_id: Annotated[
        Optional[str], Header()
    ] = None,  # This is the value of X_GITLAB_ROOT_NAMESPACE_ID_HEADER
):
    unit_primitives = [
        GitLabUnitPrimitive.COMPLETE_CODE,
        GitLabUnitPrimitive.AI_GATEWAY_MODEL_PROVIDER_PROXY,
    ]
    scopes = [
        unit_primitive
        for unit_primitive in unit_primitives
        if current_user.can(
            unit_primitive, disallowed_issuers=[CloudConnectorConfig().service_name]
        )
    ]

    if not scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to create user access token",
        )

    internal_event_client.track_event(
        f"request_{GitLabUnitPrimitive.COMPLETE_CODE}",
        category=__name__,
    )

    user_claims = current_user.claims or UserClaims()

    extra_claims = user_claims.extra or {}
    # We can only trust the provided headers when the auth token is from SaaS
    # Otherwise, a user can perform a direct call to AIGW and specify arbitrary values
    if user_claims.gitlab_realm == "saas":
        extra_claims.update(
            {
                "gitlab_project_id": x_gitlab_project_id,
                "gitlab_namespace_id": x_gitlab_namespace_id,
                "gitlab_root_namespace_id": x_gitlab_root_namespace_id,
            }
        )
    elif user_claims.gitlab_realm == "self-managed":
        extra_claims.update({"gitlab_instance_uid": user_claims.gitlab_instance_uid})

    try:
        token, expires_at = token_authority.encode(
            current_user.global_user_id,
            user_claims.gitlab_realm,
            current_user,
            user_claims.gitlab_instance_id,
            scopes=scopes,
            extra_claims=extra_claims,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate JWT",
        )

    return Token(token=token, expires_at=expires_at)
