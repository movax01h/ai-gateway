from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from gitlab_cloud_connector import GitLabFeatureCategory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.v1.prompts.invoke import PromptRequest, _invoke
from ai_gateway.async_dependency_resolver import get_prompt_registry
from ai_gateway.instrumentators.model_requests import (
    TokenUsage,
    get_token_usage,
    init_token_usage,
)
from ai_gateway.prompts.base import BasePromptRegistry
from lib.context import StarletteUser, get_current_user


class PromptResponse(BaseModel):
    content: str
    usage: TokenUsage | None = Field(
        examples=[{"model": {"input_tokens": 10, "output_tokens": 50}}]
    )


router = APIRouter()


def _process_chunk(chunk: BaseMessage):
    return PromptResponse(
        content=str(chunk.content),
        usage=get_token_usage(),
    )


@router.post(
    "/{prompt_id:path}",
    response_model=PromptResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_200_OK,
)
@feature_category(GitLabFeatureCategory.AI_ABSTRACTION_LAYER)
async def invoke(
    request: Request,  # pylint: disable=unused-argument
    prompt_request: PromptRequest,
    prompt_id: str,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
):
    init_token_usage()

    return await _invoke(
        prompt_request=prompt_request,
        prompt_id=prompt_id,
        current_user=current_user,
        prompt_registry=prompt_registry,
        process_chunk=_process_chunk,
        internal_event_category=__name__,
    )
