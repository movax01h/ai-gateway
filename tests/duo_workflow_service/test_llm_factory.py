from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from duo_workflow_service.llm_factory import VertexConfig, validate_llm_access


@pytest.mark.parametrize(
    "env_vars,expectation,calls_llm",
    [
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "hello1",
                "DUO_WORKFLOW__VERTEX_LOCATION": "key1",
            },
            does_not_raise(),
            "vertex",
        ),
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "hello2",
                "DUO_WORKFLOW__VERTEX_LOCATION": "key2",
                "ANTHROPIC_API_KEY": "anthropic-key2",
            },
            does_not_raise(),
            "vertex",
        ),
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "hello3",
                "DUO_WORKFLOW__VERTEX_LOCATION": "",
                "ANTHROPIC_API_KEY": "anthropic-key-3",
            },
            pytest.raises(RuntimeError),
            None,
        ),
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "",
                "DUO_WORKFLOW__VERTEX_LOCATION": "key4",
                "ANTHROPIC_API_KEY": "anthropic-key4",
            },
            does_not_raise(),
            "anthropic",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "anthropic-key5",
            },
            does_not_raise(),
            "anthropic",
        ),
        ({}, pytest.raises(RuntimeError), None),
    ],
)
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_validate_anthropic_variables(
    mock_anthropic_client, mock_vertex_client, env_vars, expectation, calls_llm
):
    with patch("os.environ", env_vars):
        with expectation:
            validate_llm_access()

        if calls_llm == "vertex":
            mock_anthropic_client.assert_not_called()
            mock_vertex_client.assert_called_once()
        elif calls_llm == "anthropic":
            mock_vertex_client.assert_not_called()
            mock_anthropic_client.assert_called_once()
        else:
            mock_vertex_client.assert_not_called()
            mock_anthropic_client.assert_not_called()


@pytest.mark.parametrize(
    "env_vars,calls_llm",
    [
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            "vertex",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            "anthropic",
        ),
    ],
)
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_clients_receive_max_retries_from_config(
    mock_anthropic_client,
    mock_vertex_client,
    env_vars,
    calls_llm,
):
    config = VertexConfig()
    expected_retries = config.max_retries

    with patch("os.environ", env_vars):
        validate_llm_access(config)

        if calls_llm == "vertex":
            mock_vertex_client.assert_called_once()
            assert (
                mock_vertex_client.call_args.kwargs["max_retries"] == expected_retries
            )
            mock_anthropic_client.assert_not_called()
        else:
            mock_anthropic_client.assert_called_once()
            assert (
                mock_anthropic_client.call_args.kwargs["max_retries"]
                == expected_retries
            )
            mock_vertex_client.assert_not_called()
