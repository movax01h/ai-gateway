from time import time
from typing import Annotated, AsyncIterator, Optional, Tuple

from dependency_injector.providers import Factory
from fastapi import APIRouter, Depends, HTTPException, Request, status
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
)

from ai_gateway.api.auth_utils import StarletteUser, get_current_user
from ai_gateway.api.context_utils import (
    GitLabAiRequestType,
    populate_ai_metadata_in_context,
)
from ai_gateway.api.error_utils import capture_validation_errors
from ai_gateway.api.feature_category import feature_category
from ai_gateway.api.middleware.headers import X_GITLAB_MODEL_PROMPT_CACHE_ENABLED
from ai_gateway.api.snowplow_context import get_snowplow_code_suggestion_context
from ai_gateway.api.v2.code.model_provider_handlers import (
    AnthropicHandler,
    FireworksHandler,
    LiteLlmHandler,
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
    get_code_suggestions_completions_vertex_legacy_provider,
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
    CodeCompletionsLegacy,
    CodeGenerations,
    CodeSuggestionsChunk,
)
from ai_gateway.code_suggestions.base import CodeSuggestionsOutput
from ai_gateway.code_suggestions.processing.base import ModelEngineOutput
from ai_gateway.code_suggestions.processing.ops import lang_from_filename
from ai_gateway.config import Config
from ai_gateway.instrumentators.base import TelemetryInstrumentator
from ai_gateway.internal_events import InternalEventsClient
from ai_gateway.model_metadata import ModelMetadata, create_model_metadata
from ai_gateway.models import KindModelProvider
from ai_gateway.models.base import TokensConsumptionMetadata
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.structured_logging import get_request_logger
from ai_gateway.tracking import SnowplowEvent, SnowplowEventContext
from ai_gateway.tracking.errors import log_exception
from ai_gateway.tracking.instrumentator import SnowplowInstrumentator
from lib.feature_flags.context import current_feature_flag_context

__all__ = [
    "router",
]


request_log = get_request_logger("codesuggestions")

router = APIRouter()

COMPLETIONS_AGENT_ID = "code_suggestions/completions"
GENERATIONS_AGENT_ID = "code_suggestions/generations"


async def get_prompt_registry():
    yield get_container_application().pkg_prompts.prompt_registry()


@router.post("/completions")
@router.post("/code/completions")
@feature_category(GitLabFeatureCategory.CODE_SUGGESTIONS)
@capture_validation_errors()
async def completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    current_user: Annotated[StarletteUser, Depends(get_current_user)],
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
    config: Annotated[Config, Depends(get_config)],
    completions_legacy_factory: Annotated[
        Factory[CodeCompletionsLegacy],
        Depends(get_code_suggestions_completions_vertex_legacy_provider),
    ],
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
        completions_legacy_factory,
        completions_anthropic_factory,
        completions_litellm_factory,
        completions_fireworks_factory,
        completions_agent_factory,
        completions_amazon_q_factory,
        completions_litellm_vertex_codestral_factory,
        internal_event_client,
        region=region,
    )

    snowplow_event_context = None

    populate_ai_metadata_in_context(
        model_metadata=ModelMetadata(
            name=payload.model_name or "default",
            identifier=payload.model_identifier or "default",
            provider=payload.model_provider or "default",
        ),
        feature_id=GitLabUnitPrimitive.COMPLETE_CODE.value,
        request_type=GitLabAiRequestType.COMPLETIONS,
    )

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

    populate_ai_metadata_in_context(
        model_metadata=ModelMetadata(
            name=payload.model_name or "default",
            identifier=payload.model_identifier or "default",
            provider=payload.model_provider or "default",
        ),
        feature_id=GitLabUnitPrimitive.GENERATE_CODE.value,
        request_type=GitLabAiRequestType.GENERATIONS,
    )

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
) -> CodeGenerations:
    has_model_info = (
        payload.model_name is not None and payload.model_provider is not None
    )

    if has_model_info:
        model_metadata = ModelMetadata(
            name=payload.model_name,
            endpoint=payload.model_endpoint,
            api_key=payload.model_api_key,
            provider="custom_openai",
            identifier=payload.model_identifier,
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
) -> CodeGenerations:
    if payload.prompt_id:
        return _resolve_prompt_code_generations(
            payload,
            current_user,
            prompt_registry,
            generations_agent_factory,
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
    use_llm_prompt_caching: bool,
    completions_agent_factory: Factory[CodeCompletions],
    completions_litellm_factory: Factory[CodeCompletions],
) -> CodeCompletions:
    if payload.prompt_version == 2 and not payload.prompt:
        model_metadata = ModelMetadata(
            name=payload.model_name,
            endpoint=payload.model_endpoint,
            api_key=payload.model_api_key,
            identifier=payload.model_identifier,
            provider=payload.model_provider or "text-completion-openai",
        )

        return _resolve_agent_code_completions(
            model_metadata=model_metadata,
            current_user=current_user,
            prompt_registry=prompt_registry,
            completions_agent_factory=completions_agent_factory,
        )

    return completions_litellm_factory(
        model__name=payload.model_name,
        model__endpoint=payload.model_endpoint,
        model__api_key=payload.model_api_key,
        model__provider=payload.model_provider,
        model__using_cache=use_llm_prompt_caching,
    )


def _build_code_completions(
    request: Request,
    payload: CompletionsRequestWithVersion,
    current_user: StarletteUser,
    prompt_registry: BasePromptRegistry,
    # pylint: disable=unused-argument
    completions_legacy_factory: Factory[CodeCompletionsLegacy],
    completions_anthropic_factory: Factory[CodeCompletions],
    completions_litellm_factory: Factory[CodeCompletions],
    completions_fireworks_factory: Factory[CodeCompletions],
    completions_agent_factory: Factory[CodeCompletions],
    completions_amazon_q_factory: Factory[CodeCompletions],
    completions_litellm_vertex_codestral_factory: Factory[CodeCompletions],
    internal_event_client: InternalEventsClient,
    region: str,
) -> tuple[CodeCompletions | CodeCompletionsLegacy, dict]:
    # Default to use cache
    use_llm_prompt_caching = (
        request.headers.get(X_GITLAB_MODEL_PROMPT_CACHE_ENABLED, "true") == "true"
    )
    kwargs = {}

    unit_primitive = GitLabUnitPrimitive.COMPLETE_CODE
    tracking_event = f"request_{unit_primitive}"

    if payload.model_provider == KindModelProvider.GITLAB:
        model_metadata = create_model_metadata(
            {
                "provider": KindModelProvider.GITLAB,
                "identifier": payload.model_name,
                "feature_setting": "code_completions",
            }
        )

        payload.model_name = model_metadata.identifier
        payload.model_provider = model_metadata.provider

        if model_metadata.provider == KindModelProvider.ANTHROPIC:
            AnthropicHandler(payload, request, kwargs).update_completion_params()
            code_completions = completions_anthropic_factory(
                model__name=payload.model_name,
            )
        elif model_metadata.provider == KindModelProvider.FIREWORKS:
            FireworksHandler(payload, request, kwargs).update_completion_params()
            code_completions = _resolve_code_completions_litellm(
                payload=payload,
                current_user=current_user,
                prompt_registry=prompt_registry,
                use_llm_prompt_caching=use_llm_prompt_caching,
                completions_agent_factory=completions_agent_factory,
                completions_litellm_factory=completions_fireworks_factory,
            )
        elif model_metadata.provider == KindModelProvider.VERTEX_AI:
            code_completions = _resolve_code_completions_vertex_codestral(
                payload=payload,
                completions_litellm_vertex_codestral_factory=completions_litellm_vertex_codestral_factory,
            )

            kwargs.update(
                {
                    "temperature": 0.7,
                    "max_output_tokens": 64,
                    "context_max_percent": 0.3,
                }
            )
            if payload.context:
                kwargs.update(
                    {"code_context": [ctx.content for ctx in payload.context]}
                )

    elif payload.model_provider == KindModelProvider.ANTHROPIC:
        AnthropicHandler(payload, request, kwargs).update_completion_params()
        code_completions = completions_anthropic_factory(
            model__name=payload.model_name,
        )
    elif payload.model_provider in (
        KindModelProvider.LITELLM,
        KindModelProvider.MISTRALAI,
    ):
        LiteLlmHandler(payload, request, kwargs).update_completion_params()
        code_completions = _resolve_code_completions_litellm(
            payload=payload,
            current_user=current_user,
            prompt_registry=prompt_registry,
            use_llm_prompt_caching=use_llm_prompt_caching,
            completions_agent_factory=completions_agent_factory,
            completions_litellm_factory=completions_litellm_factory,
        )

        _track_code_suggestions_event(tracking_event, internal_event_client)

        return code_completions, kwargs
    elif payload.model_provider == KindModelProvider.AMAZON_Q:
        unit_primitive = GitLabUnitPrimitive.AMAZON_Q_INTEGRATION
        tracking_event = f"request_{unit_primitive}_complete_code"
        code_completions = completions_amazon_q_factory(
            model__current_user=current_user,
            model__role_arn=payload.role_arn,
        )
    elif (
        payload.model_provider == KindModelProvider.FIREWORKS
        or not _allow_vertex_codestral(region)
    ):
        FireworksHandler(payload, request, kwargs).update_completion_params()
        code_completions = _resolve_code_completions_litellm(
            payload=payload,
            current_user=current_user,
            prompt_registry=prompt_registry,
            use_llm_prompt_caching=use_llm_prompt_caching,
            completions_agent_factory=completions_agent_factory,
            completions_litellm_factory=completions_fireworks_factory,
        )

        _track_code_suggestions_event(tracking_event, internal_event_client)

        return code_completions, kwargs
    else:
        code_completions = _resolve_code_completions_vertex_codestral(
            payload=payload,
            completions_litellm_vertex_codestral_factory=completions_litellm_vertex_codestral_factory,
        )

        # We need to pass this here since litellm.LiteLlmTextGenModel
        # sets the default temperature and max_output_tokens in the `generate` function signature
        # To override those values, the kwargs passed to `generate` is updated here
        # For further details, see:
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1172#note_2060587592 # pylint: disable=line-too-long
        #
        # The temperature value is taken from Mistral's docs: https://docs.mistral.ai/api/#operation/createFIMCompletion
        # context_max_percent is set to 0.3 to limit the amount of context right now because latency increases with
        # larger context
        kwargs.update(
            {"temperature": 0.7, "max_output_tokens": 64, "context_max_percent": 0.3}
        )
        if payload.context:
            kwargs.update({"code_context": [ctx.content for ctx in payload.context]})

    # Providers that are handled via the prompt registry perform their own UP check and event tracking. If we reach
    # this point is because we're using some other legacy provider, and we need to perform these steps now
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
) -> CodeCompletions:
    prompt = prompt_registry.get_on_behalf(
        current_user,
        COMPLETIONS_AGENT_ID,
        model_metadata=model_metadata,
        internal_event_category=__name__,
    )

    return completions_agent_factory(
        model__prompt=prompt,
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
    code_completions: Factory[CodeCompletions | CodeCompletionsLegacy],
    snowplow_event_context: Optional[SnowplowEventContext] = None,
    **kwargs: dict,
) -> any:
    with TelemetryInstrumentator().watch(payload.telemetry):
        output = await code_completions.execute(
            prefix=payload.current_file.content_above_cursor,
            suffix=payload.current_file.content_below_cursor,
            file_name=payload.current_file.file_name,
            editor_lang=payload.current_file.language_identifier,
            stream=payload.stream,
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
