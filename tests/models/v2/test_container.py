from typing import Any
from unittest import mock

import pytest

from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import _litellm_factory, _mock_selector, litellm


def test_litellm_override():
    assert "request" in litellm.module_level_aclient.event_hooks
    assert litellm.module_level_aclient.event_hooks["request"] == [log_request]


@mock.patch("ai_gateway.models.v2.container.ChatLiteLLM")
@pytest.mark.parametrize(
    ("kwargs", "expected_kwargs"),
    [
        (
            {"model": "claude-3-sonnet@20240229", "custom_llm_provider": "vertex_ai"},
            {
                "model": "claude-3-sonnet@20240229",
                "custom_llm_provider": "vertex_ai",
                "model_kwargs": {
                    "extra_headers": {
                        "anthropic-beta": "fine-grained-tool-streaming-2025-05-14,context-1m-2025-08-07"
                    },
                },
            },
        ),
        (
            {"model": "mistral-small-2503", "custom_llm_provider": "vertex_ai"},
            {
                "model": "mistral-small-2503",
                "custom_llm_provider": "vertex_ai",
            },
        ),
        (
            {
                "model": "mistral",
            },
            {
                "model": "mistral",
            },
        ),
        (
            {
                "model": "mistral",
                "disable_streaming": False,
            },
            {
                "model": "mistral",
                "disable_streaming": False,
            },
        ),
        (
            {
                "model": "mistral",
                "disable_streaming": True,
            },
            {
                "model": "mistral",
                "disable_streaming": True,
            },
        ),
    ],
)
def test_litellm_factory(
    mock_chat_lite_llm: mock.Mock,
    kwargs: dict[str, Any],
    expected_kwargs: dict[str, Any],
):
    assert _litellm_factory(**kwargs) == mock_chat_lite_llm.return_value

    mock_chat_lite_llm.assert_called_once_with(**expected_kwargs)


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
