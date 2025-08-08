from typing import Any
from unittest import mock

import pytest

from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import _litellm_factory, _mock_selector


@mock.patch("ai_gateway.models.v2.container.ChatLiteLLM")
@mock.patch("ai_gateway.models.v2.container.AsyncHTTPHandler")
@pytest.mark.parametrize(
    ("kwargs", "expected_kwargs", "expect_client_override"),
    [
        (
            {"model": "claude-3-sonnet@20240229", "custom_llm_provider": "vertex_ai"},
            {
                "model": "claude-3-sonnet@20240229",
                "custom_llm_provider": "vertex_ai",
            },
            True,
        ),
        (
            {
                "model": "mistral",
            },
            {
                "model": "mistral",
            },
            False,
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
            False,
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
            False,
        ),
    ],
)
def test_litellm_factory(
    mock_async_http_handler: mock.Mock,
    mock_chat_lite_llm: mock.Mock,
    kwargs: dict[str, Any],
    expected_kwargs: dict[str, Any],
    expect_client_override: bool,
):
    assert _litellm_factory(**kwargs) == mock_chat_lite_llm.return_value

    if expect_client_override:
        expected_kwargs["client"] = mock_async_http_handler.return_value

    mock_chat_lite_llm.assert_called_once_with(**expected_kwargs)

    if expect_client_override:
        mock_async_http_handler.assert_called_once_with(
            event_hooks={"request": [log_request]}
        )


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
