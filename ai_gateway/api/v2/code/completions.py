from dataclasses import dataclass
from time import time
from typing import Annotated, Any, AsyncIterator, Dict, Optional, Tuple

from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
    WrongUnitPrimitives,
)

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.error_utils import capture_validation_errors
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware.route import has_sufficient_usage_quota
from ai_gateway.api.snowplow_context import get_snowplow_code_suggestion_context
from ai_gateway.api.v2.code.model_provider_handlers import (
    AnthropicHandler,
    FireworksHandler,
    LiteLlmHandler,
    VertexHandler,
)
from ai_gateway.api.v2.code.typing import (
    CompletionsRequestWithVersion,
    GenerationsRequestWithVersion,
    StreamSuggestionsResponse,
    SuggestionsRequest,
    SuggestionsResponse,
)
from ai_gateway.async_dependency_resolver import (
    get_code_suggestions_completions_agent_factory_provider,
    get_code_suggestions_completions_amazon_q_factory_provider,
    get_code_suggestions_completions_anthropic_provider,
    get_code_suggestions_completions_fireworks_factory_provider,
    get_code_suggestions_completions_litellm_factory_provider,
    get_code_suggestions_completions_litellm_vertex_codestral_factory_provider,
    get_code_suggestions_generations_agent_factory_provider,
    get_code_suggestions_generations_anthropic_chat_factory_provider,
    get_code_suggestions_generations_litellm_factory_provider,
    get_code_suggestions_generations_vertex_provider,
    get_config,
    get_container_application,
    get_internal_event_client,
    get_snowplow_instrumentator,
)
from ai_gateway.code_suggestions import (
    CodeCompletions,
    CodeGenerations,
    CodeSuggestionsChunk,
)
from ai_gateway.code_suggestions.base import CodeSuggestionsOutput
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.ops import lang_from_filename
from ai_gateway.code_suggestions.processing.post.completions import (
    PostProcessor,
    PostProcessorOperation,
)
from ai_gateway.config import Config
from ai_gateway.instrumentators.base import TelemetryInstrumentator
from ai_gateway.model_metadata import ModelMetadata, create_model_metadata
from ai_gateway.models import KindModelProvider
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.structured_logging import get_request_logger
from ai_gateway.tracking import SnowplowEvent, SnowplowEventContext
from ai_gateway.tracking.errors import log_exception
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from lib.feature_flags.context import current_feature_flag_context
from lib.internal_events import InternalEventsClient
from lib.prompts.caching import X_GITLAB_MODEL_PROMPT_CACHE_ENABLED
from lib.usage_quota import UsageQuotaEvent

__all__ = [
    "router",
]


@dataclass
class CompletionConfig:
    """Configuration for code completion providers."""

    factory: Factory[CodeCompletions]
    handler_class: Optional[type] = None
    requires_prompt_registry: bool = False
    extra_kwargs: Optional[Dict[str, Any]] = None
    unit_primitive: Optional[GitLabUnitPrimitive] = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}


request_log = get_request_logger("codesuggestions")

router = APIRouter()

COMPLETIONS_AGENT_ID = "code_suggestions/completions"
GENERATIONS_AGENT_ID = "code_suggestions/generations"

LEGACY_COMPLETION_MODEL_TO_GITLAB_IDENTIFIER = {
    "vertex-ai/codestral-2501": "codestral_2501_vertex",
    "fireworks_ai/codestral-2501": "codestral_2501_fireworks",
    "anthropic/claude-sonnet-4-20250514": "claude_sonnet_4_20250514",
}


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


@router.post("/completions")
@router.post("/code/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@capture_validation_errors()
@has_sufficient_usage_quota(
    feature_qualified_name="code_suggestions",
    event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
)
async def completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    config: Annotated[Config, Depends(get_config)],
    completions_anthropic_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_anthropic_provider),
    ],
    completions_litellm_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_litellm_factory_provider),
    ],
    completions_fireworks_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_fireworks_factory_provider),
    ],
    completions_amazon_q_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_amazon_q_factory_provider),
    ],
    completions_litellm_vertex_codestral_factory: Annotated[
        Factory[CodeCompletions],
        Depends(
            get_code_suggestions_completions_litellm_vertex_codestral_factory_provider
        ),
    ],
    completions_agent_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_agent_factory_provider),
    ],
    snowplow_instrumentator: Annotated[
        SnowplowInstrumentator,
        Depends(get_snowplow_instrumentator),
    ],
    internal_event_client: Annotated[
        InternalEventsClient,
        Depends(get_internal_event_client),
    ],
):
    region = config.google_cloud_platform.location()
    code_completions, kwargs = _build_code_completions(
        request,
        payload,
        current_user,
        prompt_registry,
        completions_anthropic_factory,
        completions_litellm_factory,
        completions_fireworks_factory,
        completions_agent_factory,
        completions_amazon_q_factory,
        completions_litellm_vertex_codestral_factory,
        internal_event_client,
        region,
        config.model_keys(),
        config.model_endpoints(),
        config,
    )

    snowplow_event_context = None

    try:
        language = lang_from_filename(payload.current_file.file_name)
        language_name = language.name if language else ""
        snowplow_event_context = get_snowplow_code_suggestion_context(
            req=request,
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            language=language_name,
            global_user_id=current_user.global_user_id,
            region=region,
        )
        snowplow_instrumentator.watch(SnowplowEvent(context=snowplow_event_context))
    except Exception as e:
        log_exception(e)

    request_log.info(
        "code completion input:",
        model_name=payload.model_name,
        model_provider=payload.model_provider,
        prompt=payload.prompt if hasattr(payload, "prompt") else None,
        prefix=payload.current_file.content_above_cursor,
        suffix=payload.current_file.content_below_cursor,
        current_file_name=payload.current_file.file_name,
        stream=payload.stream,
    )

    suggestions = await _execute_code_completion(
        payload=payload,
        code_completions=code_completions,
        current_user=current_user,
        snowplow_event_context=snowplow_event_context,
        **kwargs,
    )

    if isinstance(suggestions[0], AsyncIterator):
        return await _handle_stream(suggestions[0])
    choices, tokens_consumption_metadata = _completion_suggestion_choices(suggestions)
    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=suggestions[0].model.engine,
            name=suggestions[0].model.name,
            lang=suggestions[0].lang,
            tokens_consumption_metadata=tokens_consumption_metadata,
            region=region,
        ),
        metadata=SuggestionsResponse.MetadataBase(
            enabled_feature_flags=current_feature_flag_context.get(),
        ),
        choices=choices,
    )


@router.post("/code/generations")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@capture_validation_errors()
@has_sufficient_usage_quota(
    feature_qualified_name="code_suggestions",
    event=UsageQuotaEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
)
async def generations(
    request: Request,
    payload: GenerationsRequestWithVersion,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    config: Annotated[Config, Depends(get_config)],
    generations_vertex_factory: Annotated[
        Factory[CodeGenerations],
        Depends(get_code_suggestions_generations_vertex_provider),
    ],
    generations_anthropic_chat_factory: Annotated[
        Factory[CodeGenerations],
        Depends(get_code_suggestions_generations_anthropic_chat_factory_provider),
    ],
    generations_litellm_factory: Annotated[
        Factory[CodeGenerations],
        Depends(get_code_suggestions_generations_litellm_factory_provider),
    ],
    generations_agent_factory: Annotated[
        Factory[CodeGenerations],
        Depends(get_code_suggestions_generations_agent_factory_provider),
    ],
    snowplow_instrumentator: Annotated[
        SnowplowInstrumentator, Depends(get_snowplow_instrumentator)
    ],
    internal_event_client: Annotated[
        InternalEventsClient, Depends(get_internal_event_client)
    ],
):
    if not current_user.can(
        GitLabUnitPrimitive.GENERATE_CODE,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code generations",
        )

    snowplow_event_context = None

    try:
        language = lang_from_filename(payload.current_file.file_name)
        language_name = language.name if language else ""
        snowplow_event_context = get_snowplow_code_suggestion_context(
            req=request,
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            language=language_name,
            global_user_id=current_user.global_user_id,
            region=config.google_cloud_platform.location(),
        )
        snowplow_instrumentator.watch(SnowplowEvent(context=snowplow_event_context))
    except Exception as e:
        log_exception(e)

    request_log.debug(
        "code creation input:",
        prompt=payload.prompt if hasattr(payload, "prompt") else None,
        prefix=payload.current_file.content_above_cursor,
        suffix=payload.current_file.content_below_cursor,
        current_file_name=payload.current_file.file_name,
        stream=payload.stream,
        endpoint=payload.model_endpoint,
        api_key="*" * 10 if payload.model_api_key else None,
    )

    code_generations = _build_code_generations(
        payload,
        current_user,
        prompt_registry,
        generations_vertex_factory,
        generations_anthropic_chat_factory,
        generations_litellm_factory,
        generations_agent_factory,
        internal_event_client,
        config,
    )

    if payload.prompt_version == 3:
        code_generations.with_prompt_prepared(payload.prompt)

    with TelemetryInstrumentator().watch(payload.telemetry):
        suggestion = await code_generations.execute(
            prefix=payload.current_file.content_above_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            model_provider=payload.model_provider,
            stream=payload.stream,
            snowplow_event_context=snowplow_event_context,
        )

    if isinstance(suggestion, AsyncIterator):
        return await _handle_stream(suggestion)

    request_log.debug(
        "code creation suggestion:",
        suggestion=suggestion.text,
        score=suggestion.score,
        language=suggestion.lang,
    )

    return SuggestionsResponse(
        id="id",
        created=int(time()),
        model=SuggestionsResponse.Model(
            engine=suggestion.model.engine,
            name=suggestion.model.name,
            lang=suggestion.lang,
            region=config.google_cloud_platform.location(),
        ),
        metadata=SuggestionsResponse.MetadataBase(
            enabled_feature_flags=current_feature_flag_context.get(),
        ),
        choices=_generation_suggestion_choices(suggestion.text),
    )


def _resolve_code_generations_anthropic_chat(
    payload: SuggestionsRequest,
    generations_anthropic_chat_factory: Factory[CodeGenerations],
) -> CodeGenerations:
    return generations_anthropic_chat_factory(
        model__name=payload.model_name,
        model__stop_sequences=["</new_code>"],
    )


def _resolve_prompt_code_generations(
    payload: SuggestionsRequest,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    generations_agent_factory: Factory[CodeGenerations],
    config: Config,
) -> CodeGenerations:
    has_model_info = (
        payload.model_name is not None and payload.model_provider is not None
    )

    if has_model_info:
        model_metadata = create_model_metadata(
            {
                "name": payload.model_name,
                "endpoint": payload.model_endpoint,
                "api_key": payload.model_api_key,
                "provider": "custom_openai",
                "identifier": payload.model_identifier,
            },
            mock_model_responses=config.mock_model_responses,
        )
        prompt = prompt_registry.get_on_behalf(
            current_user,
            payload.prompt_id,
            model_metadata=model_metadata,
            internal_event_category=__name__,
        )
    else:
        prompt = prompt_registry.get_on_behalf(
            current_user,
            payload.prompt_id,
            model_metadata=None,  # Don't provide model_metadata
            internal_event_category=__name__,
        )

    return generations_agent_factory(model__prompt=prompt)


def _build_code_generations(
    payload: GenerationsRequestWithVersion,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    generations_vertex_factory: Factory[CodeGenerations],
    generations_anthropic_chat_factory: Factory[CodeGenerations],
    generations_litellm_factory: Factory[CodeGenerations],
    generations_agent_factory: Factory[CodeGenerations],
    internal_event_client: InternalEventsClient,
    config: Config,
) -> CodeGenerations:
    if payload.prompt_id:
        return _resolve_prompt_code_generations(
            payload,
            current_user,
            prompt_registry,
            generations_agent_factory,
            config,
        )

    tracking_event = f"request_{GitLabUnitPrimitive.GENERATE_CODE}"

    # If we didn't use the prompt registry, we have to track the internal event manually
    _track_code_suggestions_event(tracking_event, internal_event_client)

    if payload.model_provider == KindModelProvider.ANTHROPIC:
        return _resolve_code_generations_anthropic_chat(
            payload,
            generations_anthropic_chat_factory,
        )

    if payload.model_provider == KindModelProvider.LITELLM:
        return generations_litellm_factory(
            model__name=payload.model_name,
            model__endpoint=payload.model_endpoint,
            model__api_key=payload.model_api_key,
        )

    return generations_vertex_factory()


def _resolve_code_completions_litellm(
    payload: SuggestionsRequest,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
    model_keys: dict,
    model_endpoints: dict,
    using_cache: bool,
    config: Config,
    gitlab_identifier: Optional[str] = None,
) -> CodeCompletions:
    # Use the GitLab identifier if provided (from model_provider: "gitlab")
    # Otherwise, try to map legacy provider/model_name to GitLab identifier
    if gitlab_identifier:
        name = gitlab_identifier
    else:
        name = _get_gitlab_identifier(payload.model_provider, payload.model_name)

    model_metadata = create_model_metadata(
        {
            "name": name,
            "endpoint": payload.model_endpoint,
            "api_key": payload.model_api_key,
            "identifier": payload.model_identifier,
            "provider": payload.model_provider or "text-completion-openai",
            "provider_keys": model_keys,
            "model_endpoints": model_endpoints,
            "using_cache": using_cache,
            "session_id": current_user.global_user_id,
        },
        mock_model_responses=config.mock_model_responses,
    )

    # Create post processor based on model provider and name
    post_processor = _create_post_processor_for_model(
        payload.model_provider,
        payload.model_name,
        config,
    )

    return _resolve_agent_code_completions(
        model_metadata=model_metadata,
        current_user=current_user,
        prompt_registry=prompt_registry,
        completions_agent_factory=completions_agent_factory,
        post_processor=post_processor,
    )


def _get_gitlab_identifier(model_provider: str, model_name: str) -> str:
    legacy_identifier = f"{model_provider}/{model_name}"

    return LEGACY_COMPLETION_MODEL_TO_GITLAB_IDENTIFIER.get(
        legacy_identifier, model_name
    )


def _create_post_processor_for_model(
    model_provider: str,
    model_name: str,
    config: Config,
) -> Optional[Factory]:
    """Create the appropriate post processor factory based on model provider and name."""

    # Vertex Codestral: apply STRIP_ASTERISKS
    if model_provider == KindModelProvider.VERTEX_AI and model_name == "codestral-2501":
        return Factory(
            PostProcessor,
            extras=[PostProcessorOperation.STRIP_ASTERISKS],
            exclude=config.feature_flags.excl_post_process,
        )

    # Fireworks: apply FILTER_SCORE and FIX_TRUNCATION
    if model_provider == KindModelProvider.FIREWORKS:
        return Factory(
            PostProcessor,
            exclude=config.feature_flags.excl_post_process,
            extras=[
                PostProcessorOperation.FILTER_SCORE,
                PostProcessorOperation.FIX_TRUNCATION,
            ],
            score_threshold=config.feature_flags.fireworks_score_threshold,
        )

    return None


def _get_provider_config(
    provider: KindModelProvider,
    completions_anthropic_factory: Factory[CodeCompletions],
    completions_litellm_factory: Factory[CodeCompletions],
    completions_fireworks_factory: Factory[CodeCompletions],
    completions_amazon_q_factory: Factory[CodeCompletions],
    completions_litellm_vertex_codestral_factory: Factory[CodeCompletions],
    region: str,
    payload: CompletionsRequestWithVersion,
) -> CompletionConfig:
    """Get the appropriate completion configuration for the given provider."""

    def _should_include_context(provider: KindModelProvider) -> bool:
        """Determine if this provider should include context."""

        return provider not in [KindModelProvider.ANTHROPIC, KindModelProvider.AMAZON_Q]

    def _get_context_kwargs(provider: KindModelProvider) -> Dict[str, Any]:
        """Get context kwargs if needed for this provider."""

        if _should_include_context(provider) and payload.context:
            return {"code_context": [ctx.content for ctx in payload.context]}
        return {}

    if provider == KindModelProvider.ANTHROPIC:
        return CompletionConfig(
            factory=completions_anthropic_factory,
            requires_prompt_registry=True,
            handler_class=AnthropicHandler,
            extra_kwargs=_get_context_kwargs(provider),
        )

    if provider in (KindModelProvider.LITELLM, KindModelProvider.MISTRALAI):
        return CompletionConfig(
            factory=completions_litellm_factory,
            handler_class=LiteLlmHandler,
            requires_prompt_registry=True,
            extra_kwargs=_get_context_kwargs(provider),
        )

    if provider == KindModelProvider.AMAZON_Q:
        return CompletionConfig(
            factory=completions_amazon_q_factory,
            unit_primitive=GitLabUnitPrimitive.AMAZON_Q_INTEGRATION,
            extra_kwargs=_get_context_kwargs(provider),
        )

    if provider == KindModelProvider.FIREWORKS or not _allow_vertex_codestral(region):
        return CompletionConfig(
            factory=completions_fireworks_factory,
            handler_class=FireworksHandler,
            requires_prompt_registry=True,
            extra_kwargs=_get_context_kwargs(provider),
        )

    return CompletionConfig(
        factory=completions_litellm_vertex_codestral_factory,
        handler_class=VertexHandler,
        requires_prompt_registry=True,
        extra_kwargs=_get_context_kwargs(provider),
    )


def _build_code_completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    completions_anthropic_factory: Factory[CodeCompletions],
    completions_litellm_factory: Factory[CodeCompletions],
    completions_fireworks_factory: Factory[CodeCompletions],
    completions_agent_factory: Factory[CodeCompletions],
    completions_amazon_q_factory: Factory[CodeCompletions],
    completions_litellm_vertex_codestral_factory: Factory[CodeCompletions],
    internal_event_client: InternalEventsClient,
    region: str,
    model_keys: dict,
    model_endpoints: dict,
    config: Config,
) -> tuple[CodeCompletions, dict]:
    unit_primitive = GitLabUnitPrimitive.COMPLETE_CODE
    tracking_event = f"request_{unit_primitive}"

    # Check if prompt cache is enabled via header
    using_cache = (
        request.headers.get(X_GITLAB_MODEL_PROMPT_CACHE_ENABLED, "true").lower()
        == "true"
    )

    gitlab_identifier: Optional[str] = None
    if payload.model_provider == KindModelProvider.GITLAB:
        model_metadata = create_model_metadata(
            {
                "provider": KindModelProvider.GITLAB,
                "identifier": payload.model_name,
                "feature_setting": "code_completions",
                "provider_keys": model_keys,
                "model_endpoints": model_endpoints,
            },
            mock_model_responses=config.mock_model_responses,
        )

        actual_provider = KindModelProvider.from_definition_provider(
            model_metadata.llm_definition.provider
        )
        # Store the GitLab identifier for later use
        gitlab_identifier = payload.model_name
        # Update payload with legacy model_provider/name for completions code
        payload.model_provider = actual_provider
        provider_model_name = model_metadata.llm_definition.params.get(
            "model", payload.model_name
        )
        kwargs = {}

        if actual_provider == KindModelProvider.ANTHROPIC:
            AnthropicHandler(payload, request, kwargs).update_completion_params()
        elif actual_provider == KindModelProvider.FIREWORKS:
            payload.model_name = provider_model_name
            FireworksHandler(payload, request, kwargs).update_completion_params()
        elif actual_provider == KindModelProvider.VERTEX_AI:
            payload.model_name = provider_model_name
            VertexHandler(payload, request, kwargs).update_completion_params()

    provider_config = _get_provider_config(
        payload.model_provider,
        completions_anthropic_factory,
        completions_litellm_factory,
        completions_fireworks_factory,
        completions_amazon_q_factory,
        completions_litellm_vertex_codestral_factory,
        region,
        payload,
    )

    kwargs = {}

    if provider_config.handler_class:
        provider_config.handler_class(
            payload, request, kwargs
        ).update_completion_params()

    if provider_config.requires_prompt_registry:
        code_completions = _resolve_code_completions_litellm(
            payload=payload,
            current_user=current_user,
            prompt_registry=prompt_registry,
            completions_agent_factory=completions_agent_factory,
            model_keys=model_keys,
            model_endpoints=model_endpoints,
            using_cache=using_cache,
            config=config,
            gitlab_identifier=gitlab_identifier,
        )

        _track_code_suggestions_event(tracking_event, internal_event_client)
        return code_completions, kwargs

    if payload.model_provider == KindModelProvider.AMAZON_Q:
        code_completions = provider_config.factory(
            model__current_user=current_user,
            model__role_arn=payload.role_arn,
        )
    elif payload.model_provider == KindModelProvider.ANTHROPIC:
        code_completions = provider_config.factory(model__name=payload.model_name)
    else:
        code_completions = provider_config.factory()

    kwargs.update(provider_config.extra_kwargs)

    unit_primitive = provider_config.unit_primitive or GitLabUnitPrimitive.COMPLETE_CODE
    tracking_event = f"request_{unit_primitive}"
    if provider_config.unit_primitive == GitLabUnitPrimitive.AMAZON_Q_INTEGRATION:
        tracking_event = f"request_{unit_primitive}_complete_code"
    if not current_user.can(unit_primitive):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code completions",
        )

    _track_code_suggestions_event(tracking_event, internal_event_client)

    return code_completions, kwargs


def _resolve_code_completions_vertex_codestral(
    payload: SuggestionsRequest,
    completions_litellm_vertex_codestral_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    if payload.prompt_version == 2 and payload.prompt is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot specify a prompt with the given provider and model combination",
        )

    return completions_litellm_vertex_codestral_factory()


def _resolve_agent_code_completions(
    model_metadata: ModelMetadata,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    completions_agent_factory: Factory[CodeCompletions],
    post_processor: Optional[Factory] = None,
) -> CodeCompletions:
    try:
        prompt = prompt_registry.get_on_behalf(
            current_user,
            COMPLETIONS_AGENT_ID,
            model_metadata=model_metadata,
            internal_event_category=__name__,
        )
    except WrongUnitPrimitives:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access code completions",
        )

    return completions_agent_factory(
        model__prompt=prompt,
        post_processor=post_processor,
    )


def _completion_suggestion_choices(
    suggestions: list,
) -> Tuple[list[SuggestionsResponse.Choice], Optional[TokensConsumptionMetadata]]:
    if len(suggestions) == 0:
        return [], None
    choices: list[SuggestionsResponse.Choice] = []

    choices = []
    tokens_consumption_metadata = None
    for suggestion in suggestions:
        request_log.debug(
            "code completion suggestion:",
            suggestion=suggestion.text,
            score=suggestion.score,
            language=suggestion.lang,
        )
        if not suggestion.text:
            continue

        if tokens_consumption_metadata is None:
            # We take the first metadata from the suggestions since they are all the same
            if isinstance(suggestion, ModelEngineOutput):
                tokens_consumption_metadata = suggestion.tokens_consumption_metadata
            elif isinstance(suggestion, CodeSuggestionsOutput) and suggestion.metadata:
                tokens_consumption_metadata = (
                    suggestion.metadata.tokens_consumption_metadata
                )

        choices.append(
            SuggestionsResponse.Choice(
                text=suggestion.text,
            )
        )
    return choices, tokens_consumption_metadata


def _generation_suggestion_choices(text: str) -> list:
    return [SuggestionsResponse.Choice(text=text)] if text else []


def _allow_vertex_codestral(region: str):
    return not region.startswith("asia-")


async def _handle_stream(
    response: AsyncIterator[CodeSuggestionsChunk],
) -> StreamSuggestionsResponse:
    async def _stream_generator():
        async for result in response:
            yield result.text

    return StreamSuggestionsResponse(
        _stream_generator(), media_type="text/event-stream"
    )


async def _execute_code_completion(
    payload: CompletionsRequestWithVersion,
    code_completions: CodeCompletions,
    current_user: StarletteUser,
    snowplow_event_context: Optional[SnowplowEventContext] = None,
    **kwargs: dict,
) -> any:
    with TelemetryInstrumentator().watch(payload.telemetry):
        output = await code_completions.execute(
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            stream=payload.stream or False,  # Ensure stream is bool, not None
            user=current_user.cloud_connector_user,  # Pass the underlying CloudConnectorUser
            snowplow_event_context=snowplow_event_context,
            **kwargs,
        )

    if isinstance(code_completions, CodeCompletions):
        return [output]
    return output


def _track_code_suggestions_event(
    tracking_event: str, internal_event_client: InternalEventsClient
):
    internal_event_client.track_event(
        tracking_event,
        category=__name__,
    )
