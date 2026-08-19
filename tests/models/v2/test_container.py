from unittest.mock import MagicMock, patch

import pytest
from google.genai.client import Client

from ai_gateway.model_selection.models import CompletionType
from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import (
    ContainerModels,
    _compute_fireworks_allowed_api_bases,
    _mock_selector,
    litellm,
)


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


def test_lite_llm_chat_uses_configured_request_timeout():
    """The DWS ChatLiteLLM factory must set request_timeout from duo_chat config so litellm enforces a real deadline
    instead of falling back to its default (~600s) timeout."""
    container = ContainerModels()
    container.config.from_dict(
        {
            "custom_models": {"enabled": False},
            "bedrock_guardrail_config": None,
            "fireworks_api_base_url": "",
            "duo_chat": {"model_request_timeout": 45.0},
            "mock_model_responses": False,
            "use_agentic_mock": False,
        }
    )

    model = container.lite_llm_chat_fn(model="claude-sonnet-4-5")

    assert model.request_timeout == 45.0


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
            "custom_models": {"enabled": False},
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
