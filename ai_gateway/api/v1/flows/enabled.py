from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from gitlab_cloud_connector import GitLabFeatureCategory, GitLabUnitPrimitive
from pydantic import BaseModel

from ai_gateway.api.feature_category import feature_category
from ai_gateway.config import get_config
from lib.context import StarletteUser, get_current_user
from lib.flow_enablement import enabled_gated_flows

__all__ = ["router"]

router = APIRouter()


class EnabledFlowsResponse(BaseModel):
    enabled_flows: list[str]


@router.get("/enabled", response_model=EnabledFlowsResponse)
@feature_category(GitLabFeatureCategory.DUO_AGENT_PLATFORM)
async def enabled(
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
) -> EnabledFlowsResponse:
    """List the gated flows enabled for the caller's own instance; other flows are always enabled."""
    if not current_user.can(GitLabUnitPrimitive.DUO_AGENT_PLATFORM):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to list enabled flows",
        )

    return EnabledFlowsResponse(
        enabled_flows=enabled_gated_flows(
            current_user, gitlab_issuer=get_config().gitlab_url
        )
    )
