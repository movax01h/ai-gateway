import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google.genai.client import Client
from langchain_openai.decisions import Choice

from ai_gateway.model_selection.models import CompletionType
from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import (
    ContainerModels,
    _compute_fireworks_allowed_api_bases,
    _mock_selector,
    litellm,
)
from ai_gateway.models.v2.openai_decisions import decision_response


def test_litellm_override():
    assert "request" in litellm.module_level_aclient.event_hooks
    assert litellm.module_level_aclient.event_hooks["request"] == [log_request]


@pytest.mark.parametrize(
    ("mock_model_responses", "use_agentic_mock", "expected_selector"),
    [
        (False, False, "original"),
        (True, False, "mocked"),
        (True, True, "agentic"),
        (
            False,
            True,
            "original",
        ),  # use_agentic_mock has no effect when mock_model_responses is False
    ],
)
def test_mock_selector(mock_model_responses, use_agentic_mock, expected_selector):
    result = _mock_selector(mock_model_responses, use_agentic_mock)
    assert result == expected_selector


@pytest.mark.parametrize(
    ("fireworks_api_base_url", "expected"),
    [
        (
            "https://api.fireworks.ai/inference/v1",
            frozenset(["https://api.fireworks.ai/inference/v1"]),
        ),
        # trailing slash is stripped during normalization
        (
            "https://api.fireworks.ai/inference/v1/",
            frozenset(["https://api.fireworks.ai/inference/v1"]),
        ),
        # empty base URL is excluded
        ("", frozenset()),
        ("   ", frozenset()),
    ],
)
def test_compute_fireworks_allowed_api_bases(fireworks_api_base_url, expected):
    result = _compute_fireworks_allowed_api_bases(fireworks_api_base_url)
    assert result == expected


@pytest.mark.parametrize(
    ("duo_workflow_dict", "expected_url"),
    [
        (
            {
                "use_caching_proxy": False,
                "caching_proxy": {"url": "http://localhost:8888"},
            },
            None,
        ),
        (
            {
                "use_caching_proxy": True,
                "caching_proxy": {"url": "http://proxy.test:8888"},
            },
            "http://proxy.test:8888",
        ),
    ],
)
def test_duo_workflow_caching_proxy_url_via_container(duo_workflow_dict, expected_url):
    container = ContainerModels()
    container.config.from_dict(
        {
            "duo_workflow": duo_workflow_dict,
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    assert container._duo_workflow().caching_proxy_url() == expected_url


@pytest.mark.parametrize(
    ("factory_name", "timeout_attr"),
    [
        ("lite_llm_chat_fn", "request_timeout"),
        ("anthropic_claude_chat_fn", "default_request_timeout"),
        ("openai_chat_fn", "request_timeout"),
        ("google_chat_gen_vertex_ai_global_fn", "timeout"),
    ],
)
def test_chat_factory_uses_configured_request_timeout(
    factory_name, timeout_attr, monkeypatch
):
    """Every chat factory must set its timeout attribute from duo_chat config so each model class enforces a real
    deadline on a single request instead of falling back to its own (or no) default."""
    # `openai_chat_fn` passes no api_key; langchain_openai requires one (or the env var) at construction time.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    container = ContainerModels()
    container.config.from_dict(
        {
            "custom_models": {"enabled": False, "user_id_header": None},
            "bedrock_guardrail_config": None,
            "fireworks_api_base_url": "",
            "duo_chat": {"model_request_timeout": 45.0},
            "duo_workflow": {"use_caching_proxy": False},
            "google_cloud_platform": {"project": "test-project"},
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    with patch(
        "ai_gateway.models.v2.container.connect_google_gen_vertex_ai",
        side_effect=lambda *_args, **_kwargs: MagicMock(spec=Client),
    ):
        model = getattr(container, factory_name)(model="claude-sonnet-4-5")

    assert getattr(model, timeout_attr) == 45.0


@pytest.mark.parametrize(
    ("vertexai_location", "runway_region", "expected"),
    [
        (None, "europe-west9", "europe-west4"),
        (None, "us-east4", "us-central1"),
        ("europe-west1", "us-east4", "europe-west1"),
    ],
)
def test_lite_llm_completion_resolves_vertex_location(
    vertexai_location, runway_region, expected
):
    container = ContainerModels()
    container.config.from_dict(
        {
            "custom_models": {"enabled": False, "user_id_header": None},
            "bedrock_guardrail_config": None,
            "fireworks_api_base_url": "",
            "vertex_text_model": {"location": runway_region},
            "vertexai_location": vertexai_location,
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    model = container.lite_llm_completion_fn(
        model="codestral-2", completion_type=CompletionType.TEXT
    )

    assert model.vertex_location == expected


@pytest.mark.parametrize("vertexai_location", [None, "europe-west2"])
def test_lite_llm_embedding_uses_vertexai_location(vertexai_location):
    container = ContainerModels()
    container.config.from_dict(
        {
            "custom_models": {"enabled": False, "user_id_header": None},
            "vertexai_location": vertexai_location,
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    model = container.lite_llm_embedding_fn(model="text-embedding-005")

    assert model.vertex_location == vertexai_location


@pytest.mark.asyncio
async def test_google_chat_gen_vertex_ai_client_not_shared_across_resolutions():
    """Each resolution of the Vertex Gemini factory must build its own `google.genai.Client` instead of sharing one
    process-wide singleton.

    `ChatGoogleGenerativeAI.__del__` (langchain_google_genai) closes whatever `client` it was given as soon as that
    resolution's wrapper is garbage collected. A shared `Client` previously let any one request's wrapper being
    collected close the client out from under another in-flight request still using it, crashing it with
    `assert self._connector is not None` deep in aiohttp.
    """
    container = ContainerModels()
    container.config.from_dict(
        {
            "custom_models": {"enabled": False},
            "duo_chat": {"model_request_timeout": 45.0},
            "google_cloud_platform": {"project": "test-project"},
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    with patch(
        "ai_gateway.models.v2.container.connect_google_gen_vertex_ai",
        side_effect=lambda *_args, **_kwargs: MagicMock(spec=Client),
    ) as mock_connect:
        model_a = container.google_chat_gen_vertex_ai_global_fn(model="gemini-2.5-pro")
        model_b = container.google_chat_gen_vertex_ai_global_fn(model="gemini-2.5-pro")

    assert mock_connect.call_count == 2
    assert model_a.client is not model_b.client

    await container.google_gen_vertex_ai_http_client().aclose()


@pytest.mark.asyncio
async def test_google_chat_gen_vertex_ai_http_client_shared_across_resolutions():
    """The underlying http client (and its connection pool) is shared across every resolution, unlike the `Client`
    wrapper itself, so that Vertex Gemini requests reuse pooled connections instead of paying a fresh TCP+TLS handshake
    per call."""
    container = ContainerModels()
    container.config.from_dict(
        {
            "custom_models": {"enabled": False},
            "duo_chat": {"model_request_timeout": 45.0},
            "google_cloud_platform": {"project": "test-project"},
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    with patch(
        "ai_gateway.models.v2.container.connect_google_gen_vertex_ai",
        side_effect=lambda *_args, **kwargs: MagicMock(
            spec=Client, http_client=kwargs.get("http_client")
        ),
    ):
        model_a = container.google_chat_gen_vertex_ai_global_fn(model="gemini-2.5-pro")
        model_b = container.google_chat_gen_vertex_ai_global_fn(model="gemini-2.5-pro")

    assert model_a.client.http_client is model_b.client.http_client
    assert model_a.client.http_client is container.google_gen_vertex_ai_http_client()

    await container.google_gen_vertex_ai_http_client().aclose()


@pytest.mark.asyncio
async def test_openai_decisions_fn_answers_questions():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
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
                    }
                ],
            },
        )

    container = ContainerModels()
    container.config.from_dict(
        {"decision": {"request_timeout": 10.0, "max_retries": 0}}
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        decisions = container.openai_decisions_fn(
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            http_async_client=client,
        )
        response = await decisions.ainvoke(
            {
                "input": "Fix the typo in the README",
                "questions": {
                    "tier": Choice(
                        instructions="Which model tier does this request need?",
                        choices={"small": "A small change.", "large": "Many files."},
                    )
                },
            }
        )

    assert decision_response(response).choices["tier"].choice == "small"
    assert str(requests[0].url) == "https://api.openai.com/v1/decisions"
    assert json.loads(requests[0].content)["model"] == decisions.model


@pytest.mark.asyncio
async def test_openai_decisions_fn_instances_share_one_http_client():
    container = ContainerModels()
    container.config.from_dict(
        {"decision": {"request_timeout": 10.0, "max_retries": 0}}
    )

    first = container.openai_decisions_fn(api_key="test-key")
    second = container.openai_decisions_fn(api_key="test-key")

    assert first is not second
    assert first.http_async_client is second.http_async_client
    assert first.http_async_client is container.http_async_client_openai()

    await container.http_async_client_openai().aclose()


@pytest.mark.parametrize(
    ("request_timeout", "max_retries"),
    [
        (10.0, 0),
        (2.5, 1),
    ],
)
def test_openai_decisions_fn_uses_configured_timeout_and_retries(
    request_timeout, max_retries
):
    """A decision classifies so a caller can fall back, so the factory wires a short timeout and no client retries from
    config instead of the OpenAI SDK defaults (a ~600s read timeout and two retries)."""
    container = ContainerModels()
    container.config.from_dict(
        {"decision": {"request_timeout": request_timeout, "max_retries": max_retries}}
    )

    model = container.openai_decisions_fn(api_key="[REDACTED]")

    assert model.timeout == request_timeout
    assert model.max_retries == max_retries
