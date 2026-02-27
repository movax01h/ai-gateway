"""Tests for ai_gateway.prompts.config.base module."""

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.model_selection.models import CompletionLiteLLMParams, CompletionType
from ai_gateway.prompts.config.base import InMemoryPromptConfig, PromptConfig


class TestInMemoryPromptConfig:
    """Test InMemoryPromptConfig class functionality."""

    def test_inmemory_prompt_config_creation(self):
        """Test that guards stable API of InMemoryPromptConfig.

        The InMemoryPromptConfig is exposed via gRPC public API, therefore any change to its structure must be backward
        compatible!
        """
        config_data = {
            "prompt_id": "test/prompt",
            "name": "test_prompt",
            "model": {
                "params": {
                    "model": "claude-3-sonnet",
                }
            },
            "unit_primitives": [GitLabUnitPrimitive.DUO_CHAT],
            "prompt_template": {
                "system": "You are a helpful assistant.",
                "user": "{{user_input}}",
            },
        }

        config = InMemoryPromptConfig(**config_data)

        assert config.prompt_id == "test/prompt"
        assert config.name == "test_prompt"
        assert config.model.params.model == "claude-3-sonnet"
        assert config.unit_primitives == [GitLabUnitPrimitive.DUO_CHAT]
        assert config.prompt_template == {
            "system": "You are a helpful assistant.",
            "user": "{{user_input}}",
        }

    @pytest.mark.parametrize(
        "unit_primitives,expected_unit_primitive",
        [
            ([GitLabUnitPrimitive.DUO_CHAT], GitLabUnitPrimitive.DUO_CHAT),
            ([], GitLabUnitPrimitive.DUO_AGENT_PLATFORM),
        ],
    )
    def test_to_prompt_data(
        self,
        unit_primitives: list[GitLabUnitPrimitive],
        expected_unit_primitive: GitLabUnitPrimitive,
    ):
        """Test converting InMemoryPromptConfig to PromptConfig."""
        config_data = {
            "prompt_id": "test/prompt",
            "name": "test_prompt",
            "model": {
                "params": {
                    "model": "claude-3-sonnet",
                }
            },
            "unit_primitives": unit_primitives,
            "prompt_template": {
                "system": "You are a helpful assistant.",
                "user": "{{user_input}}",
            },
            "params": {
                "stop": ["Human:", "Assistant:"],
                "timeout": 30.0,
            },
        }

        in_memory_config = InMemoryPromptConfig.model_validate(config_data)
        prompt_data = in_memory_config.to_prompt_data()

        assert isinstance(prompt_data, dict)

        prompt_config = PromptConfig(**prompt_data)

        # Should be a PromptConfig instance
        assert isinstance(prompt_config, PromptConfig)
        assert not isinstance(prompt_config, InMemoryPromptConfig)

        # Should have all relevant fields
        assert prompt_config.name == "test_prompt"
        assert prompt_config.model.params.model == "claude-3-sonnet"
        assert prompt_config.unit_primitive == expected_unit_primitive
        assert prompt_config.prompt_template == {
            "system": "You are a helpful assistant.",
            "user": "{{user_input}}",
        }
        assert prompt_config.params
        assert prompt_config.params.stop == ["Human:", "Assistant:"]
        assert prompt_config.params.timeout == 30.0

        # Should not have excluded fields
        assert not hasattr(prompt_config, "prompt_id")
        assert not hasattr(prompt_config, "unit_primitives")


class TestCompletionLiteLLMParams:
    def test_validate_fim_format_requires_format(self):
        with pytest.raises(ValueError, match="fim_format is required"):
            CompletionLiteLLMParams(
                model="codestral-2501",
                completion_type=CompletionType.FIM,
            )

    def test_validate_fim_format_accepts_format(self):
        params = CompletionLiteLLMParams(
            model="codestral-2501",
            completion_type=CompletionType.FIM,
            fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
        )

        assert params.fim_format == "</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]"
