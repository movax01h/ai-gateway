from time import time
from typing import Annotated, AsyncIterator, Optional

from dependency_injector.providers import Factory
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
)

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware import X_GITLAB_LANGUAGE_SERVER_VERSION
from ai_gateway.api.snowplow_context import get_snowplow_code_suggestion_context
from ai_gateway.api.v3.code.typing import (
    CodeContextPayload,
    CodeEditorComponents,
    CompletionRequest,
    CompletionResponse,
    EditorContentCompletionPayload,
    EditorContentGenerationPayload,
    ModelMetadata,
    ResponseMetadataBase,
    StreamHandler,
    StreamModelEngine,
    StreamSuggestionsResponse,
)
from ai_gateway.async_dependency_resolver import get_config, get_container_application
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
    LanguageServerVersion,
    ModelProvider,
)
from ai_gateway.code_suggestions.base import SAAS_PROMPT_MODEL_MAP
from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.feature_flags.context import current_feature_flag_context
from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.models import KindModelProvider
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.structured_logging import get_request_logger
from ai_gateway.tracking import SnowplowEventContext

__all__ = [
    "router",
    "code_suggestions",
]

request_log = get_request_logger("codesuggestions")

router = APIRouter()


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


async def handle_stream(
    stream: AsyncIterator[CodeSuggestionsChunk],
    metadata: ResponseMetadataBase,
) -> StreamSuggestionsResponse:
    async def _stream_response_generator():
        async for chunk in stream:
            yield chunk.text

    return StreamSuggestionsResponse(
        _stream_response_generator(), media_type="text/event-stream"
    )


@router.post("/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
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


# This function is also used by `v4/code/suggestions`. When making
# changes, ensure you consider its effects on both v3 and v4.
async def code_suggestions(
    request: Request,
    payload: CompletionRequest,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    config: Config,
    stream_handler: StreamHandler = handle_stream,
):
    language_server_version = LanguageServerVersion.from_string(
        request.headers.get(X_GITLAB_LANGUAGE_SERVER_VERSION, None)
    )
    component = payload.prompt_components[0]
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
            code_context=code_context,
            stream_handler=stream_handler,
            snowplow_event_context=snowplow_code_suggestion_context,
            model_metadata=payload.model_metadata,
        )
    if component.type == CodeEditorComponents.GENERATION:
        return await code_generation(
            current_user=current_user,
            payload=component.payload,
            code_context=code_context,
            prompt_registry=prompt_registry,
            stream_handler=stream_handler,
            snowplow_event_context=snowplow_code_suggestion_context,
            model_metadata=payload.model_metadata,
        )


@inject
async def code_completion(
    payload: EditorContentCompletionPayload,
    current_user: StarletteUser,
    stream_handler: StreamHandler,
    snowplow_event_context: SnowplowEventContext,
    completions_legacy_factory: Factory[CodeCompletionsLegacy] = Provide[
        ContainerApplication.code_suggestions.completions.vertex_legacy.provider
    ],
    completions_anthropic_factory: Factory[CodeCompletions] = Provide[
        ContainerApplication.code_suggestions.completions.anthropic.provider
    ],
    completions_amazon_q_factory: Factory[CodeCompletions] = Provide[
        ContainerApplication.code_suggestions.completions.amazon_q_factory.provider
    ],
    code_context: Optional[list[CodeContextPayload]] = None,
    model_metadata: TypeModelMetadata = None,
):
    kwargs = {}

    if payload.model_provider == ModelProvider.ANTHROPIC:
        # TODO: As we migrate to v3 we can rewrite this to use prompt registry
        engine = completions_anthropic_factory(model__name=payload.model_name)
        kwargs.update({"raw_prompt": payload.prompt})
    elif payload.model_provider == KindModelProvider.AMAZON_Q or (
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
        engine = completions_legacy_factory()

    if payload.choices_count > 0:
        kwargs.update({"candidate_count": payload.choices_count})

    suggestions = await engine.execute(
        prefix=payload.content_above_cursor,
        suffix=payload.content_below_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        stream=payload.stream,
        code_context=code_context,
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
                engine=suggestions[0].model.engine,
                name=suggestions[0].model.name,
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
    payload: EditorContentGenerationPayload,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    stream_handler: StreamHandler,
    snowplow_event_context: SnowplowEventContext,
    generations_vertex_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.vertex.provider
    ],
    generations_anthropic_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.anthropic_default.provider
    ],
    agent_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.agent_factory.provider
    ],
    generations_amazon_q_factory: Factory[CodeGenerations] = Provide[
        ContainerApplication.code_suggestions.generations.amazon_q_factory.provider
    ],
    code_context: Optional[list[CodeContextPayload]] = None,
    model_metadata: TypeModelMetadata = None,
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
        # TODO: Since we are migrating to prompt registry, we should sunset this branch
        if model_provider == KindModelProvider.ANTHROPIC:
            engine = generations_anthropic_factory()
        else:
            engine = generations_vertex_factory()

        if payload.prompt:
            engine.with_prompt_prepared(payload.prompt)

    suggestion = await engine.execute(
        prefix=payload.content_above_cursor,
        file_name=payload.file_name,
        editor_lang=payload.language_identifier,
        model_provider=model_provider,
        stream=payload.stream,
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
                engine=suggestion.model.engine,
                name=suggestion.model.name,
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
