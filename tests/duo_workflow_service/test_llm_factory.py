from contextlib import nullcontext as does_not_raise
from unittest.mock import Mock, patch

import pytest

from ai_gateway.models import KindAnthropicModel
from duo_workflow_service.llm_factory import (
    AnthropicConfig,
    AnthropicStopReason,
    VertexConfig,
    create_chat_model,
    validate_llm_access,
)


@pytest.mark.parametrize(
    "env_vars,expectation,calls_llm",
    [
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            does_not_raise(),
            "vertex",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            does_not_raise(),
            "anthropic",  # Falls back to Anthropic and succeeds
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-proj",
            },
            pytest.raises(
                RuntimeError,
                match="ANTHROPIC_API_KEY needs to be set for Anthropic provider",
            ),
            None,
        ),
        (
            {},
            pytest.raises(
                RuntimeError,
                match="ANTHROPIC_API_KEY needs to be set for Anthropic provider",
            ),  # Falls back to Anthropic but no API key
            None,
        ),
    ],
)
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_validate_anthropic_variables(
    mock_anthropic_client,
    mock_vertex_client,
    env_vars,
    expectation,
    calls_llm,
):
    # Mock the invoke method to return a response
    mock_response = Mock()
    mock_response.content = "I am Claude, an AI assistant."
    mock_anthropic_client.return_value.invoke.return_value = mock_response
    mock_vertex_client.return_value.invoke.return_value = mock_response

    with patch("os.environ", env_vars):
        with expectation:
            validate_llm_access()

        if calls_llm == "vertex":
            mock_anthropic_client.assert_not_called()
            mock_vertex_client.assert_called_once()

            call_kwargs = mock_vertex_client.call_args.kwargs
            assert (
                call_kwargs["model_name"]
                == KindAnthropicModel.CLAUDE_SONNET_4_VERTEX.value
            )
            assert call_kwargs["project"] == "test-proj"
            assert call_kwargs["location"] == "test-loc"
        elif calls_llm == "anthropic":
            mock_vertex_client.assert_not_called()
            mock_anthropic_client.assert_called_once()

            call_kwargs = mock_anthropic_client.call_args.kwargs
            assert call_kwargs["model_name"] == KindAnthropicModel.CLAUDE_SONNET_4.value
            assert call_kwargs["betas"] == [
                "extended-cache-ttl-2025-04-11",
                "context-1m-2025-08-07",
            ]
        else:
            mock_vertex_client.assert_not_called()
            mock_anthropic_client.assert_not_called()


@pytest.mark.parametrize(
    "env_vars,config_class,model_name,calls_llm",
    [
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            VertexConfig,
            None,
            "vertex",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            AnthropicConfig,
            "claude-sonnet-4-20250514",  # Required for AnthropicConfig
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
    config_class,
    model_name,
    calls_llm,
):
    # Mock the invoke method to return a response
    mock_response = Mock()
    mock_response.content = "I am Claude, an AI assistant."
    mock_anthropic_client.return_value.invoke.return_value = mock_response
    mock_vertex_client.return_value.invoke.return_value = mock_response

    with patch("os.environ", env_vars):
        # Create the appropriate config based on the test case
        if config_class == VertexConfig:
            config = VertexConfig()
        else:
            config = AnthropicConfig(model_name=model_name)

        expected_retries = config.max_retries

        # Use validate_llm_access with the config
        validate_llm_access(config)

        if calls_llm == "vertex":
            mock_vertex_client.assert_called_once()
            assert (
                mock_vertex_client.call_args.kwargs["max_retries"] == expected_retries
            )
            mock_anthropic_client.assert_not_called()
        else:
            mock_anthropic_client.assert_called_once()
            call_kwargs = mock_anthropic_client.call_args.kwargs
            assert call_kwargs["max_retries"] == expected_retries
            assert call_kwargs["betas"] == [
                "extended-cache-ttl-2025-04-11",
                "context-1m-2025-08-07",
            ]
            mock_vertex_client.assert_not_called()


@pytest.mark.parametrize(
    "env_vars,config_type,model_param,expected_model,calls_llm",
    [
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            "vertex",
            "custom-model-name",
            "custom-model-name",
            "vertex",
        ),
        (
            {
                "AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            "vertex",
            None,
            "claude-sonnet-4@20250514",  # Default when no feature flags
            "vertex",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            "anthropic",
            "claude-3-7-sonnet-20250219",
            "claude-3-7-sonnet-20250219",
            "anthropic",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            "anthropic",
            None,
            None,  # Will fail validation if None
            "anthropic",
        ),
    ],
)
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_new_chat_client_with_custom_model(
    mock_anthropic_client,
    mock_vertex_client,
    env_vars,
    config_type,
    model_param,
    expected_model,
    calls_llm,
):
    with patch("os.environ", env_vars):
        if config_type == "vertex":
            if model_param:
                config = VertexConfig(model_name=model_param)
            else:
                config = VertexConfig()
        else:  # anthropic
            if model_param:
                config = AnthropicConfig(model_name=model_param)
            else:
                # This should raise validation error since model_name is required
                # and must be a valid KindAnthropicModel value
                with pytest.raises(ValueError):
                    config = AnthropicConfig()
                return

        create_chat_model(config=config)

        if calls_llm == "vertex":
            mock_vertex_client.assert_called_once()
            assert mock_vertex_client.call_args.kwargs["model_name"] == expected_model
            assert mock_vertex_client.call_args.kwargs["project"] == "test-proj"
            assert mock_vertex_client.call_args.kwargs["location"] == "test-loc"
            mock_anthropic_client.assert_not_called()
        else:
            mock_anthropic_client.assert_called_once()
            call_kwargs = mock_anthropic_client.call_args.kwargs
            assert call_kwargs["model_name"] == expected_model
            assert call_kwargs["betas"] == [
                "extended-cache-ttl-2025-04-11",
                "context-1m-2025-08-07",
            ]
            mock_vertex_client.assert_not_called()


class TestAnthropicStopReason:
    """Test cases for AnthropicStopReason enum."""

    def test_enum_values(self):
        """Test that all enum values are correctly defined."""
        assert AnthropicStopReason.END_TURN == "end_turn"
        assert AnthropicStopReason.MAX_TOKENS == "max_tokens"
        assert AnthropicStopReason.STOP_SEQUENCE == "stop_sequence"
        assert AnthropicStopReason.TOOL_USE == "tool_use"
        assert AnthropicStopReason.PAUSE_TURN == "pause_turn"
        assert AnthropicStopReason.REFUSAL == "refusal"

    def test_values_class_method(self):
        """Test that values() returns all enum values as a list."""
        expected_values = [
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
        actual_values = AnthropicStopReason.values()

        assert isinstance(actual_values, list)
        assert len(actual_values) == 6
        assert set(actual_values) == set(expected_values)

    def test_abnormal_values_class_method(self):
        """Test that abnormal_values() returns only abnormal stop reasons."""
        expected_abnormal = ["max_tokens", "refusal"]
        actual_abnormal = AnthropicStopReason.abnormal_values()

        assert isinstance(actual_abnormal, list)
        assert len(actual_abnormal) == 2
        assert set(actual_abnormal) == set(expected_abnormal)

    def test_abnormal_values_subset_of_all_values(self):
        """Test that abnormal values are a subset of all values."""
        all_values = set(AnthropicStopReason.values())
        abnormal_values = set(AnthropicStopReason.abnormal_values())

        assert abnormal_values.issubset(all_values)
