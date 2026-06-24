from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, Request
from gitlab_cloud_connector import GitLabFeatureCategory

from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from ai_gateway.api.v3.code.typing import (
    CodeEditorComponents,
    CompletionRequest,
    ResponseMetadataBase,
    StreamSuggestionsResponse,
)
from ai_gateway.async_dependency_resolver import get_config, get_container_application
from ai_gateway.code_suggestions import CodeSuggestionsChunk
from ai_gateway.code_suggestions.handler import code_suggestions as _code_suggestions
from ai_gateway.config import Config
from ai_gateway.prompts import BasePromptRegistry
from lib.context import StarletteUser, get_current_user
from lib.events import FeatureQualifiedNameStatic
from lib.usage_quota import UsageQuotaEvent

__all__ = [
    "code_suggestions",
    "router",
]

router = APIRouter()


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


async def handle_stream(
    stream: AsyncIterator[CodeSuggestionsChunk],
    metadata: ResponseMetadataBase,  # pylint: disable=unused-argument
) -> StreamSuggestionsResponse:
    async def _stream_response_generator():
        async for chunk in stream:
            yield chunk.text

    return StreamSuggestionsResponse(
        _stream_response_generator(), media_type="text/event-stream"
    )


async def get_event_type(payload: CompletionRequest) -> str:
    """Determine event type from completion request payload."""
    if not payload.prompt_components:
        raise ValueError("No prompt components provided")
    component = payload.prompt_components[0]
    return (
        UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS
        if component.type == CodeEditorComponents.COMPLETION
        else UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS
    )


@router.post("/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@has_sufficient_usage_quota(
    feature_qualified_name=FeatureQualifiedNameStatic.CODE_SUGGESTIONS,
    event=get_event_type,
)
async def completions(
    request: Request,
    payload: CompletionRequest,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    config: Annotated[Config, Depends(get_config)],
):
    return await code_suggestions(
        request=request,
        payload=payload,
        current_user=current_user,
        prompt_registry=prompt_registry,
        config=config,
    )


async def code_suggestions(
    request: Request,
    payload: CompletionRequest,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    config: Config,
):
    return await _code_suggestions(
        request=request,
        payload=payload,
        current_user=current_user,
        prompt_registry=prompt_registry,
        config=config,
        stream_handler=handle_stream,
    )
