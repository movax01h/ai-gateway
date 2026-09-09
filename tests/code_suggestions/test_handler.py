from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dependency_injector import providers
from gitlab_cloud_connector import WrongUnitPrimitives
from starlette_context import request_cycle_context

from ai_gateway.code_suggestions import handler as handler_module
from ai_gateway.code_suggestions.handler import code_completion


@pytest.fixture(name="suggestion")
def suggestion_fixture():
    return SimpleNamespace(
        text="x = 1",
        score=0,
        lang="python",
        model_metadata=SimpleNamespace(
            engine="fireworks_ai", name="codestral_2508_fireworks"
        ),
        metadata=SimpleNamespace(tokens_consumption_metadata=None),
    )


@pytest.fixture(name="payload")
def payload_fixture():
    payload = MagicMock()
    payload.model_provider = None
    payload.role_arn = None
    payload.content_above_cursor = "def f("
    payload.content_below_cursor = ""
    payload.file_name = "test.py"
    payload.language_identifier = "python"
    payload.stream = False
    return payload


@pytest.fixture(name="generation_payload")
def generation_payload_fixture(payload):
    payload.prompt_id = None
    payload.prompt_enhancer = None
    return payload


@pytest.fixture(name="engine")
def engine_fixture(suggestion):
    engine = MagicMock()
    engine.execute = AsyncMock(return_value=suggestion)
    return engine


@pytest.fixture(name="agent_factory")
def agent_factory_fixture(engine):
    return MagicMock(return_value=engine)


@pytest.fixture(name="prompt_registry")
def prompt_registry_fixture():
    return MagicMock()


@pytest.fixture(name="snowplow_instrumentator")
def snowplow_instrumentator_fixture(mock_ai_gateway_container):
    instrumentator = MagicMock()
    mock_ai_gateway_container.snowplow.instrumentator.override(
        providers.Object(instrumentator)
    )
    yield instrumentator
    mock_ai_gateway_container.snowplow.instrumentator.reset_override()


def assert_snowplow_event(instrumentator, expected_context):
    instrumentator.watch.assert_called_once()
    event = instrumentator.watch.call_args.args[0]
    assert isinstance(event, handler_module.SnowplowEvent)
    assert event.context is expected_context


@pytest.mark.asyncio
async def test_code_completion_applies_model_driven_behavior(
    payload, engine, agent_factory, prompt_registry
):
    """A completion: wires the model's post-processor and the resolved model
    metadata to the engine, forwards the model's context cap to execute, and
    reports the resolved model under metadata.model (the shared response shape)."""
    model_metadata = MagicMock()
    model_metadata.provider = "fireworks_ai"

    prompt = MagicMock()
    prompt_registry.get_on_behalf.return_value = prompt

    snowplow_event_context = MagicMock()
    snowplow_event_context.region = "us-central1"

    post_processor = object()

    with (
        patch.object(
            handler_module,
            "create_post_processor_for_model_metadata",
            return_value=post_processor,
        ),
        patch.object(
            handler_module,
            "completion_context_max_percent_for_model_metadata",
            return_value=0.3,
        ),
    ):
        response = await code_completion(
            payload=payload,
            current_user=MagicMock(),
            prompt_registry=prompt_registry,
            stream_handler=AsyncMock(),
            snowplow_event_context=snowplow_event_context,
            completions_agent_factory=agent_factory,
            completions_amazon_q_factory=MagicMock(),
            model_metadata=model_metadata,
            config=MagicMock(),
        )

    agent_factory.assert_called_once_with(
        model__prompt=prompt,
        post_processor=post_processor,
        model_metadata=model_metadata,
    )
    assert engine.execute.await_args.kwargs.get("context_max_percent") == 0.3

    assert response.metadata.model.engine == "fireworks_ai"
    assert response.metadata.model.name == "codestral_2508_fireworks"


@pytest.mark.asyncio
async def test_code_suggestions_dispatches_completion_without_event(
    snowplow_instrumentator,
):
    component = MagicMock()
    component.type = handler_module.CodeEditorComponents.COMPLETION
    payload = MagicMock()
    payload.prompt_components = [component]

    request = MagicMock()
    request.headers = {}
    current_user = MagicMock()
    current_user.can.return_value = True
    snowplow_context = MagicMock()

    with (
        request_cycle_context({}),
        patch.object(handler_module, "CloudConnectorConfig"),
        patch.object(
            handler_module,
            "get_snowplow_code_suggestion_context",
            return_value=snowplow_context,
        ),
        patch.object(
            handler_module, "code_completion", new=AsyncMock()
        ) as completion_mock,
    ):
        await handler_module.code_suggestions(
            request=request,
            payload=payload,
            current_user=current_user,
            prompt_registry=MagicMock(),
            config=MagicMock(),
            stream_handler=AsyncMock(),
        )

    snowplow_instrumentator.watch.assert_not_called()
    completion_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_code_completion_watches_event_after_authorization(
    payload, engine, prompt_registry, snowplow_instrumentator
):
    model_metadata = MagicMock()
    model_metadata.provider = "openai"
    snowplow_event_context = MagicMock()

    with (
        patch.object(
            handler_module,
            "create_post_processor_for_model_metadata",
            return_value=None,
        ),
        patch.object(
            handler_module,
            "completion_context_max_percent_for_model_metadata",
            return_value=None,
        ),
    ):
        await code_completion(
            payload=payload,
            current_user=MagicMock(),
            prompt_registry=prompt_registry,
            stream_handler=AsyncMock(),
            snowplow_event_context=snowplow_event_context,
            completions_agent_factory=MagicMock(return_value=engine),
            completions_amazon_q_factory=MagicMock(),
            model_metadata=model_metadata,
            config=MagicMock(),
        )

    assert_snowplow_event(snowplow_instrumentator, snowplow_event_context)


@pytest.mark.asyncio
async def test_code_completion_skips_event_when_unauthorized(
    payload, prompt_registry, snowplow_instrumentator
):
    model_metadata = MagicMock()
    model_metadata.provider = "openai"
    prompt_registry.get_on_behalf.side_effect = WrongUnitPrimitives

    with pytest.raises(handler_module.HTTPException):
        await code_completion(
            payload=payload,
            current_user=MagicMock(),
            prompt_registry=prompt_registry,
            stream_handler=AsyncMock(),
            snowplow_event_context=MagicMock(),
            completions_agent_factory=MagicMock(),
            completions_amazon_q_factory=MagicMock(),
            model_metadata=model_metadata,
            config=MagicMock(),
        )

    snowplow_instrumentator.watch.assert_not_called()


@pytest.mark.asyncio
async def test_code_completion_swallows_snowplow_errors(
    payload, engine, prompt_registry, snowplow_instrumentator
):
    model_metadata = MagicMock()
    model_metadata.provider = "openai"
    snowplow_instrumentator.watch.side_effect = RuntimeError("snowplow down")

    with (
        patch.object(
            handler_module,
            "create_post_processor_for_model_metadata",
            return_value=None,
        ),
        patch.object(
            handler_module,
            "completion_context_max_percent_for_model_metadata",
            return_value=None,
        ),
        patch.object(handler_module, "log_exception") as log_mock,
    ):
        response = await code_completion(
            payload=payload,
            current_user=MagicMock(),
            prompt_registry=prompt_registry,
            stream_handler=AsyncMock(),
            snowplow_event_context=MagicMock(),
            completions_agent_factory=MagicMock(return_value=engine),
            completions_amazon_q_factory=MagicMock(),
            model_metadata=model_metadata,
            config=MagicMock(),
        )

    log_mock.assert_called_once()
    assert response.metadata.model.name == "codestral_2508_fireworks"


@pytest.mark.asyncio
async def test_code_generation_watches_event_after_authorization(
    generation_payload, agent_factory, prompt_registry, snowplow_instrumentator
):
    snowplow_event_context = MagicMock()

    await handler_module.code_generation(
        payload=generation_payload,
        current_user=MagicMock(),
        prompt_registry=prompt_registry,
        stream_handler=AsyncMock(),
        snowplow_event_context=snowplow_event_context,
        agent_factory=agent_factory,
        generations_amazon_q_factory=MagicMock(),
    )

    assert_snowplow_event(snowplow_instrumentator, snowplow_event_context)


@pytest.mark.asyncio
async def test_code_generation_skips_event_when_unauthorized(
    generation_payload, prompt_registry, snowplow_instrumentator
):
    prompt_registry.get_on_behalf.side_effect = WrongUnitPrimitives

    with pytest.raises(WrongUnitPrimitives):
        await handler_module.code_generation(
            payload=generation_payload,
            current_user=MagicMock(),
            prompt_registry=prompt_registry,
            stream_handler=AsyncMock(),
            snowplow_event_context=MagicMock(),
            agent_factory=MagicMock(),
            generations_amazon_q_factory=MagicMock(),
        )

    snowplow_instrumentator.watch.assert_not_called()
