from typing import Any
from unittest import mock

import pytest

from ai_gateway.models.base import log_request
from ai_gateway.models.v2.container import _litellm_factory


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
                "model_kwargs": {"stream_options": {"include_usage": True}},
            },
            True,
        ),
        (
            {"model": "mistral"},
            {
                "model": "mistral",
                "model_kwargs": {"stream_options": {"include_usage": True}},
            },
            False,
        ),
        (
            {
                "model": "mistral",
                "model_kwargs": {
                    "another_model_kwarg": "my_value",
                    "stream_options": {"another_stream_option": 1},
                },
            },
            {
                "model": "mistral",
                "model_kwargs": {
                    "another_model_kwarg": "my_value",
                    "stream_options": {
                        "another_stream_option": 1,
                        "include_usage": True,
                    },
                },
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
                "model_kwargs": {
                    "stream_options": {
                        "include_usage": True,
                    },
                },
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
