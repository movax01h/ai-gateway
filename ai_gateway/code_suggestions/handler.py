from enum import StrEnum
from time import time
from typing import Any, AsyncIterator, List, Optional, Protocol, Union

from dependency_injector.providers import Factory
from dependency_injector.wiring import Provide, inject
from fastapi import HTTPException, Request, status
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    GitLabUnitPrimitive,
    WrongUnitPrimitives,
)
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.responses import StreamingResponse
from starlette_context import context as starlette_context

from ai_gateway.api.middleware import X_GITLAB_LANGUAGE_SERVER_VERSION
from ai_gateway.api.snowplow_context import get_snowplow_code_suggestion_context
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeGenerations,
    CodeSuggestionsChunk,
    LanguageServerVersion,
)
from ai_gateway.code_suggestions.base import SAAS_PROMPT_MODEL_MAP
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.model_metadata import (
    TypeModelMetadata,
    build_default_code_completions_metadata,
    create_model_metadata,
)
from ai_gateway.models import KindModelProvider
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.structured_logging import get_request_logger
from ai_gateway.tracking import SnowplowEventContext
from lib.context import StarletteUser, current_model_metadata_context
from lib.feature_flags.context import current_feature_flag_context
from lib.prompts.caching import X_GITLAB_MODEL_PROMPT_CACHE_ENABLED

__all__ = [
    "CodeEditorComponents",
    "CompletionResponse",
    "ModelMetadata",
    "ResponseMetadataBase",
    "StreamHandler",
    "StreamModelEngine",
    "StreamSuggestionsResponse",
    "code_completion",
    "code_generation",
    "code_suggestions",
]


request_log = get_request_logger("codesuggestions")


class CodeEditorComponents(StrEnum):
    COMPLETION = "code_editor_completion"
    GENERATION = "code_editor_generation"
    CONTEXT = "code_context"


class ModelMetadata(BaseModel):
    engine: Optional[str] = None
    name: Optional[str] = None
    lang: Optional[str] = None


class ResponseMetadataBase(BaseModel):
    model: Optional[ModelMetadata] = None
    timestamp: int
    enabled_feature_flags: Optional[list[str]] = None
    region: Optional[str] = None


class CompletionResponse(BaseModel):
    class Choice(BaseModel):
        text: str
        index: int = 0
        finish_reason: str = "length"

    choices: list[Choice]
    metadata: Optional[ResponseMetadataBase] = None


class StreamSuggestionsResponse(StreamingResponse):
    pass


StreamModelEngine = Union[CodeCompletions, CodeGenerations]


class StreamHandler(Protocol):
    async def __call__(
        self,
        stream: AsyncIterator[CodeSuggestionsChunk],
        metadata: ResponseMetadataBase,
    ) -> Union[StreamSuggestionsResponse, EventSourceResponse]:
        pass


async def code_suggestions(
    request: Request,
    payload: Any,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    config: Config,
    stream_handler: StreamHandler,
):
    language_server_version = LanguageServerVersion.from_string(
        request.headers.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
    )
    using_cache = (
        request.headers.get(X_GITLAB_MODEL_PROMPT_CACHE_ENABLED, "true").lower()
        == "true"
    )
    component = payload.prompt_components[0]
    starlette_context["code_suggestion_type"] = component.type.value
    code_context = [
        component.payload.content
        for component in payload.prompt_components
        if component.type == CodeEditorComponents.CONTEXT
        and language_server_version.supports_advanced_context()
    ] or None

    snowplow_code_suggestion_context = get_snowplow_code_suggestion_context(
        req=request,
        prefix=component.payload.content_above_cursor,
        suffix=component.payload.content_below_cursor,
        language=component.payload.language_identifier,
        global_user_id=current_user.global_user_id,
        region=config.google_cloud_platform.location(),
    )

    if component.type == CodeEditorComponents.COMPLETION:
        if not current_user.can(
            GitLabUnitPrimitive.COMPLETE_CODE,
            disallowed_issuers=[CloudConnectorConfig().service_name],
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized to access code suggestions",
            )

        return await code_completion(
            payload=component.payload,
            current_user=current_user,
            prompt_registry=prompt_registry,
            code_context=code_context,
            stream_handler=stream_handler,
            snowplow_event_context=snowplow_code_suggestion_context,
            model_metadata=current_model_metadata_context.get(),
            config=config,
            using_cache=using_cache,
        )
    if component.type == CodeEditorComponents.GENERATION:
        return await code_generation(
            current_user=current_user,
            payload=component.payload,
            code_context=code_context,
            prompt_registry=prompt_registry,
            stream_handler=stream_handler,
            snowplow_event_context=snowplow_code_suggestion_context,
            model_metadata=current_model_metadata_context.get(),
            config=config,
        )


@inject
async def code_completion(
    payload: Any,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    stream_handler: StreamHandler,
    snowplow_event_context: SnowplowEventContext,
    completions_agent_factory: Factory[CodeCompletions] = Provide[
        ContainerApplication.code_suggestions.completions.agent_factory.provider
    ],
    completions_amazon_q_factory: Factory[CodeCompletions] = Provide[
        ContainerApplication.code_suggestions.completions.amazon_q_factory.provider
    ],
    code_context: Optional[List[Any]] = None,
    model_metadata: TypeModelMetadata = None,
    config: Optional[Config] = None,
    using_cache: bool = True,
):
    kwargs = {}

    if payload.model_provider == KindModelProvider.AMAZON_Q or (
        model_metadata and model_metadata.provider == KindModelProvider.AMAZON_Q
    ):
        if not current_user.can(
            GitLabUnitPrimitive.AMAZON_Q_INTEGRATION,
            disallowed_issuers=[CloudConnectorConfig().service_name],
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized to access code suggestions",
            )

        engine = completions_amazon_q_factory(
            model__current_user=current_user,
            model__role_arn=payload.role_arn or model_metadata.role_arn,
        )
    else:
        if model_metadata is None:
            if config is None:
                raise ValueError(
                    "config must be provided when model_metadata is not set"
                )
            model_metadata = build_default_code_completions_metadata(
                fireworks_api_base_url=config.fireworks_api_base_url(),
                model_keys=config.model_keys(),
                user=current_user,
                using_cache=using_cache,
                mock_model_responses=config.mock_model_responses,
            )

        try:
            prompt = prompt_registry.get_on_behalf(
                current_user,
                "code_suggestions/completions",
                model_metadata=model_metadata,
                internal_event_category=__name__,
            )
        except WrongUnitPrimitives:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized to access code suggestions",
            )
        engine = completions_agent_factory(model__prompt=prompt)

    suggestions = await engine.execute(
        prefix=payload.content_above_cursor,
        suffix=payload.content_below_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        stream=payload.stream,
        code_context=code_context,
        user=current_user.cloud_connector_user,
        snowplow_event_context=snowplow_event_context,
        **kwargs,
    )

    if not isinstance(suggestions, list):
        suggestions = [suggestions]

    if isinstance(suggestions[0], AsyncIterator):
        stream_metadata = _get_stream_metadata(engine, snowplow_event_context)
        return await stream_handler(suggestions[0], stream_metadata)

    return CompletionResponse(
        choices=_completion_suggestion_choices(suggestions),
        metadata=ResponseMetadataBase(
            timestamp=int(time()),
            model=ModelMetadata(
                engine=suggestions[0].model_metadata.engine,
                name=suggestions[0].model_metadata.name,
                lang=suggestions[0].lang,
            ),
            enabled_feature_flags=current_feature_flag_context.get(),
        ),
    )


def _completion_suggestion_choices(suggestions: list) -> list:
    if len(suggestions) == 0:
        return []

    choices = []
    for suggestion in suggestions:
        request_log.debug(
            "code completion suggestion:",
            suggestion=suggestion,
            score=suggestion.score,
            language=suggestion.lang,
        )

        if not suggestion.text:
            continue

        choices.append(CompletionResponse.Choice(text=suggestion.text))

    return choices


@inject
async def code_generation(
    payload: Any,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    stream_handler: StreamHandler,
    snowplow_event_context: SnowplowEventContext,
    agent_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.agent_factory.provider
    ],
    generations_amazon_q_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.amazon_q_factory.provider
    ],
    # pylint: disable=unused-argument
    code_context: Optional[List[Any]] = None,
    model_metadata: Optional[TypeModelMetadata] = None,
    config: Optional[Config] = None,
):
    model_provider = payload.model_provider or (
        model_metadata and model_metadata.provider
    )
    if model_provider == KindModelProvider.AMAZON_Q:
        if not current_user.can(
            GitLabUnitPrimitive.AMAZON_Q_INTEGRATION,
            disallowed_issuers=[CloudConnectorConfig().service_name],
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized to access code suggestions",
            )

        engine = generations_amazon_q_factory(
            model__current_user=current_user,
            model__role_arn=payload.role_arn or model_metadata.role_arn,
        )
    elif payload.prompt_id:
        # for backward compatibility, eventually prmpt_version should be a mandatory field
        prompt_version = payload.prompt_version or "^1.0.0"
        # For SaaS: prompt_version and prompt_id are mandatory fields
        # in case prompt_id is present, model_provider is not directly passed in from request
        model_provider = SAAS_PROMPT_MODEL_MAP[prompt_version]["model_provider"]

        prompt = prompt_registry.get_on_behalf(
            user=current_user,
            prompt_id=payload.prompt_id,
            prompt_version=payload.prompt_version,
            model_metadata=model_metadata,
            internal_event_category=__name__,
        )
        engine = agent_factory(model__prompt=prompt)

        request_log.info(
            "Executing code generation with prompt registry",
            prompt_name=prompt.name,
            prompt_model_class=prompt.model.__class__.__name__,
            prompt_model_name=prompt.model_name,
        )
    else:
        # If model_provider is specified in payload but no prompt_id, use it to override model_metadata
        if model_provider and not model_metadata:
            if model_provider == KindModelProvider.ANTHROPIC:
                model_metadata = create_model_metadata(
                    {"provider": "gitlab", "identifier": "claude_sonnet_4_5_20250929"},
                    mock_model_responses=(
                        config.mock_model_responses if config else False
                    ),
                )
            elif model_provider == KindModelProvider.VERTEX_AI:
                model_metadata = create_model_metadata(
                    {
                        "provider": "gitlab",
                        "identifier": "claude_sonnet_4_5_20250929_vertex",
                    },
                    mock_model_responses=(
                        config.mock_model_responses if config else False
                    ),
                )

        prompt = prompt_registry.get_on_behalf(
            user=current_user,
            prompt_id="code_suggestions/generations",
            model_metadata=model_metadata,
            internal_event_category=__name__,
        )
        engine = agent_factory(model__prompt=prompt)

        request_log.info(
            "Executing code generation with prompt registry (legacy path)",
            prompt_name=prompt.name,
            prompt_model_class=prompt.model.__class__.__name__,
            prompt_model_name=prompt.model_name,
        )

    suggestion = await engine.execute(
        prefix=payload.content_above_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        model_provider=model_provider,
        stream=payload.stream,
        user=current_user.cloud_connector_user,
        snowplow_event_context=snowplow_event_context,
        prompt_enhancer=payload.prompt_enhancer,
        suffix=payload.content_below_cursor,
    )

    if isinstance(suggestion, AsyncIterator):
        stream_metadata = _get_stream_metadata(engine, snowplow_event_context)
        return await stream_handler(suggestion, stream_metadata)

    choices = (
        [CompletionResponse.Choice(text=suggestion.text)] if suggestion.text else []
    )

    return CompletionResponse(
        choices=choices,
        metadata=ResponseMetadataBase(
            timestamp=int(time()),
            model=ModelMetadata(
                engine=suggestion.model_metadata.engine,
                name=suggestion.model_metadata.name,
                lang=suggestion.lang,
            ),
            enabled_feature_flags=current_feature_flag_context.get(),
        ),
    )


def _get_stream_metadata(
    engine: StreamModelEngine,
    snowplow_event_context: SnowplowEventContext,
) -> ResponseMetadataBase:
    return ResponseMetadataBase(
        timestamp=int(time()),
        model=ModelMetadata(
            engine=engine.model.metadata.engine,
            name=engine.model.metadata.name,
        ),
        enabled_feature_flags=current_feature_flag_context.get(),
        region=snowplow_event_context.region,
    )
