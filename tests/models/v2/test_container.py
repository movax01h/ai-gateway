import pytest

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
