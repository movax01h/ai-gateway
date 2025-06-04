# pylint: disable=file-naming-for-tests

from contextlib import nullcontext as does_not_raise
from unittest.mock import Mock, patch

import pytest

from ai_gateway.models import KindAnthropicModel
from duo_workflow_service.llm_factory import (
    AnthropicConfig,
    VertexConfig,
    create_chat_model,
    validate_llm_access,
)


@pytest.mark.parametrize(
    "env_vars,expectation,calls_llm",
    [
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "test-proj",
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
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "test-proj",
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
@patch("duo_workflow_service.llm_factory.current_feature_flag_context")
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_validate_anthropic_variables(
    mock_anthropic_client,
    mock_vertex_client,
    mock_feature_flag_context,
    env_vars,
    expectation,
    calls_llm,
):
    mock_feature_flag_context.get.return_value = set()

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
                == KindAnthropicModel.CLAUDE_3_5_SONNET_V2_VERTEX.value
            )
            assert call_kwargs["project"] == "test-proj"
            assert call_kwargs["location"] == "test-loc"
        elif calls_llm == "anthropic":
            mock_vertex_client.assert_not_called()
            mock_anthropic_client.assert_called_once()

            call_kwargs = mock_anthropic_client.call_args.kwargs
            assert (
                call_kwargs["model_name"] == KindAnthropicModel.CLAUDE_3_7_SONNET.value
            )
        else:
            mock_vertex_client.assert_not_called()
            mock_anthropic_client.assert_not_called()


@pytest.mark.parametrize(
    "env_vars,config_class,model_name,calls_llm",
    [
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "test-proj",
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
            "claude-3-5-sonnet-20241022",  # Required for AnthropicConfig
            "anthropic",
        ),
    ],
)
@patch("duo_workflow_service.llm_factory.current_feature_flag_context")
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_clients_receive_max_retries_from_config(
    mock_anthropic_client,
    mock_vertex_client,
    mock_feature_flag_context,
    env_vars,
    config_class,
    model_name,
    calls_llm,
):
    # Mock feature flags to return empty set
    mock_feature_flag_context.get.return_value = set()

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
            assert (
                mock_anthropic_client.call_args.kwargs["max_retries"]
                == expected_retries
            )
            mock_vertex_client.assert_not_called()


@pytest.mark.parametrize(
    "env_vars,config_type,model_param,expected_model,calls_llm",
    [
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            "vertex",
            "custom-model-name",
            "custom-model-name",
            "vertex",
        ),
        (
            {
                "DUO_WORKFLOW__VERTEX_PROJECT_ID": "test-proj",
                "DUO_WORKFLOW__VERTEX_LOCATION": "test-loc",
            },
            "vertex",
            None,
            "claude-3-5-sonnet-v2@20241022",  # Default when no feature flags
            "vertex",
        ),
        (
            {
                "ANTHROPIC_API_KEY": "test-key",
            },
            "anthropic",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20241022",
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
@patch("duo_workflow_service.llm_factory.current_feature_flag_context")
@patch("duo_workflow_service.llm_factory.ChatAnthropicVertex")
@patch("duo_workflow_service.llm_factory.ChatAnthropic")
def test_new_chat_client_with_custom_model(
    mock_anthropic_client,
    mock_vertex_client,
    mock_feature_flag_context,
    env_vars,
    config_type,
    model_param,
    expected_model,
    calls_llm,
):
    # Mock feature flags to return an empty set
    mock_feature_flag_context.get.return_value = set()

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
            assert (
                mock_anthropic_client.call_args.kwargs["model_name"] == expected_model
            )
            mock_vertex_client.assert_not_called()
