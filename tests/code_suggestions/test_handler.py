from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_gateway.code_suggestions import handler as handler_module
from ai_gateway.code_suggestions.handler import code_completion


def _make_payload():
    payload = MagicMock()
    payload.model_provider = None
    payload.role_arn = None
    payload.content_above_cursor = "def f("
    payload.content_below_cursor = ""
    payload.file_name = "test.py"
    payload.language_identifier = "python"
    payload.stream = False
    return payload


def _make_suggestion():
    return SimpleNamespace(
        text="x = 1",
        score=0,
        lang="python",
        model_metadata=SimpleNamespace(
            engine="fireworks_ai", name="codestral_2508_fireworks"
        ),
        metadata=SimpleNamespace(tokens_consumption_metadata=None),
    )


@pytest.mark.asyncio
async def test_code_completion_applies_model_driven_behavior():
    """A completion: wires the model's post-processor and the resolved model
    metadata to the engine, forwards the model's context cap to execute, and
    reports the resolved model under metadata.model (the shared response shape)."""
    model_metadata = MagicMock()
    model_metadata.provider = "fireworks_ai"

    prompt = MagicMock()
    prompt_registry = MagicMock()
    prompt_registry.get_on_behalf.return_value = prompt

    engine = MagicMock()
    engine.execute = AsyncMock(return_value=_make_suggestion())
    agent_factory = MagicMock(return_value=engine)

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
            payload=_make_payload(),
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
