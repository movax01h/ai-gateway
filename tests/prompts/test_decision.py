"""A decision prompt records its usage through the standard Prompt pipeline."""

import json
from typing import Any
from unittest import mock
from unittest.mock import Mock

import httpx
import pytest
from langchain_core.messages import HumanMessage
from langchain_openai.decisions import Choice
from openai import APIConnectionError
from pydantic import SecretStr

from ai_gateway.model_selection.models import BaseModelParams, ModelClassProvider
from ai_gateway.models.v2.openai_decisions import (
    OpenAIDecisionsModel,
    decision_response,
)
from ai_gateway.prompts.base import Prompt
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.decision import decision_prompt_template_factory
from lib.billing_events.service import LLMOperationType
from lib.context import get_llm_operations, init_llm_operations
from lib.internal_events.client import InternalEventsClient

_RESPONSE_BODY = {
    "model": "gpt-6-luna",
    "answers": [
        {
            "type": "choice",
            "name": "category",
            "choice": "billing",
            "probabilities": [
                {"value": "billing", "probability": 0.9},
                {"value": "technical", "probability": 0.1},
            ],
            "confidence": 0.8,
        }
    ],
    "usage": {"input_tokens": 120, "output_tokens": 0},
}

_QUESTIONS = {
    "category": Choice(
        instructions="Which team should handle this?",
        choices={"billing": "Payments and invoices.", "technical": "Product problems."},
    )
}

_INPUT = {"request": "I was charged twice for my order.", "questions": _QUESTIONS}


def _config() -> PromptConfig:
    return PromptConfig(
        name="Classify request",
        # The chat params of the shared gpt_6_luna entry reach the decision model too.
        model=ModelConfig(
            params=BaseModelParams(model="gpt-6-luna", max_tokens=128_000)
        ),
        unit_primitive="duo_agent_platform",
        operation_type=LLMOperationType.DECISION,
        prompt_template={"user": "{{request}}"},
    )


@pytest.fixture(name="requests_sent")
def requests_sent_fixture() -> list[httpx.Request]:
    return []


@pytest.fixture(name="internal_event_client")
def internal_event_client_fixture():
    return Mock(spec=InternalEventsClient)


@pytest.fixture(name="prompt")
def prompt_fixture(internal_event_client: Mock, requests_sent: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        requests_sent.append(request)
        return httpx.Response(200, json=_RESPONSE_BODY)

    def model_factory(*, model: str, **kwargs: Any) -> OpenAIDecisionsModel:
        return OpenAIDecisionsModel(
            model=model,
            api_key=SecretStr("test-key"),
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            **kwargs,
        )

    return Prompt(
        ModelClassProvider.OPENAI,
        model_factory,
        _config(),
        prompt_template_factory=decision_prompt_template_factory,
        internal_event_client=internal_event_client,
        internal_event_extra={"workflow_id": "123", "agent_name": "router"},
    )


def test_template_renders_the_input_and_passes_the_questions():
    template = decision_prompt_template_factory(ModelClassProvider.OPENAI, _config())

    assert template.invoke(_INPUT) == {
        "input": [HumanMessage("I was charged twice for my order.")],
        "questions": _QUESTIONS,
    }


@pytest.mark.asyncio
async def test_sends_the_rendered_input_and_returns_the_answers(
    prompt: Prompt, requests_sent: list[httpx.Request]
):
    message = await prompt.ainvoke(_INPUT)

    assert decision_response(message).choices["category"].choice == "billing"
    body = json.loads(requests_sent[0].content)
    assert "I was charged twice for my order." in json.dumps(body["input"])
    assert body["questions"][0]["name"] == "category"


@pytest.mark.asyncio
async def test_a_failed_decision_is_not_retried(internal_event_client: Mock):
    requests_sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_sent.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    def model_factory(*, model: str, **kwargs: Any) -> OpenAIDecisionsModel:
        return OpenAIDecisionsModel(
            model=model,
            api_key=SecretStr("test-key"),
            http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            # Count only the Prompt's own attempts, not the OpenAI client's.
            **{**kwargs, "max_retries": 0},
        )

    prompt = Prompt(
        ModelClassProvider.OPENAI,
        model_factory,
        _config(),
        prompt_template_factory=decision_prompt_template_factory,
        internal_event_client=internal_event_client,
    )

    with mock.patch("asyncio.sleep") as sleep, pytest.raises(APIConnectionError):
        await prompt.ainvoke(_INPUT)

    assert len(requests_sent) == 1
    sleep.assert_not_called()


@pytest.mark.asyncio
async def test_records_a_decision_billing_entry(prompt: Prompt):
    init_llm_operations()

    await prompt.ainvoke(_INPUT)

    assert get_llm_operations() == [
        {
            "token_count": 120,
            "model_id": "gpt-6-luna",
            "model_engine": "openai",
            "model_provider": "openai",
            "model_vendor": "openai",
            "prompt_tokens": 120,
            "completion_tokens": 0,
            "agent_name": "router",
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "operation_type": "decision",
        }
    ]


@pytest.mark.asyncio
async def test_marks_the_token_usage_event_as_a_decision(
    prompt: Prompt, internal_event_client: Mock
):
    await prompt.ainvoke(_INPUT)

    internal_event_client.track_event.assert_called_once()
    event_name = internal_event_client.track_event.call_args.args[0]
    extra = internal_event_client.track_event.call_args.kwargs[
        "additional_properties"
    ].extra
    assert event_name == "token_usage_duo_agent_platform"
    assert extra["operation_type"] == "decision"
    assert extra["workflow_id"] == "123"
