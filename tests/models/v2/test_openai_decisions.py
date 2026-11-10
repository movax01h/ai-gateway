import asyncio
from typing import Any
from unittest.mock import patch

import httpx
import openai
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_openai.decisions import Choice, Predicate
from pydantic import SecretStr

from ai_gateway.models.v2.openai_decisions import (
    OpenAIDecisionsModel,
    decision_response,
)

_RESPONSE_BODY = {
    "model": "gpt-6-luna",
    "answers": [
        {
            "type": "choice",
            "name": "tier",
            "choice": "small",
            "probabilities": [
                {"value": "small", "probability": 0.9},
                {"value": "large", "probability": 0.1},
            ],
            "confidence": 0.8,
        },
        {"type": "predicate", "name": "urgent", "probability": 0.2},
    ],
    "usage": {"input_tokens": 120, "output_tokens": 0},
}

_REQUEST = {
    "input": "Fix the typo in the README",
    "questions": {
        "tier": Choice(
            instructions="Which model tier does this request need?",
            choices={"small": "A small change.", "large": "Many files."},
        ),
        "urgent": Predicate(instructions="Is this request urgent?"),
    },
}


class _ErrorRecorder(BaseCallbackHandler):
    def __init__(self) -> None:
        self.errors: list[BaseException] = []

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.errors.append(error)


def _model(status_code: int = 200, body: Any = None) -> OpenAIDecisionsModel:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code, json=_RESPONSE_BODY if body is None else body
        )

    return OpenAIDecisionsModel(
        model="gpt-6-luna",
        api_key=SecretStr("test-key"),
        max_retries=0,
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_reports_usage_to_the_usage_callback():
    usage_cb = UsageMetadataCallbackHandler()

    await _model().ainvoke(_REQUEST, {"callbacks": [usage_cb]})

    assert usage_cb.usage_metadata == {
        "gpt-6-luna": {"input_tokens": 120, "output_tokens": 0, "total_tokens": 120}
    }


@pytest.mark.asyncio
async def test_returns_the_answers():
    message = await _model().ainvoke(_REQUEST)

    response = decision_response(message)
    assert response.choices["tier"].choice == "small"
    assert response.predicates["urgent"].probability == 0.2


@pytest.mark.asyncio
async def test_missing_usage_counts_as_zero():
    usage_cb = UsageMetadataCallbackHandler()
    body = {key: value for key, value in _RESPONSE_BODY.items() if key != "usage"}

    await _model(body=body).ainvoke(_REQUEST, {"callbacks": [usage_cb]})

    assert usage_cb.usage_metadata["gpt-6-luna"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_errors_end_the_llm_run():
    recorder = _ErrorRecorder()

    with pytest.raises(openai.InternalServerError):
        await _model(status_code=500, body={"error": {"message": "boom"}}).ainvoke(
            _REQUEST, {"callbacks": [recorder]}
        )

    assert len(recorder.errors) == 1


@pytest.mark.asyncio
async def test_cancellation_ends_the_llm_run_and_propagates():
    """A cancelled call (or an `asyncio.wait_for` deadline) must still end the LLM run in an error.

    `asyncio.CancelledError` subclasses `BaseException`, not `Exception`, so an
    `except Exception` block would skip `run.on_llm_error`. The model must catch
    it, report the error, and re-raise so cancellation semantics are preserved.
    """
    recorder = _ErrorRecorder()
    model = _model()

    with (
        patch.object(model, "_adecide", side_effect=asyncio.CancelledError),
        pytest.raises(asyncio.CancelledError),
    ):
        await model.ainvoke(_REQUEST, {"callbacks": [recorder]})

    assert len(recorder.errors) == 1
    assert isinstance(recorder.errors[0], asyncio.CancelledError)


def test_satisfies_the_prompt_model_interface():
    model = _model()

    assert model._llm_type == "openai_decisions"
    assert model._identifying_params == {"model": "gpt-6-luna"}
    assert model.disable_streaming is True


def test_ignores_chat_params_from_the_shared_model_entry():
    model = OpenAIDecisionsModel(
        model="gpt-6-luna", api_key=SecretStr("test-key"), max_tokens=128_000
    )

    assert not hasattr(model, "max_tokens")
