"""Tests for ai_gateway.prompts.config.base module."""

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive

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
                    "model_class_provider": "anthropic",
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

    def test_to_prompt_config(self):
        """Test converting InMemoryPromptConfig to PromptConfig."""
        config_data = {
            "prompt_id": "test/prompt",
            "name": "test_prompt",
            "model": {
                "params": {
                    "model": "claude-3-sonnet",
                    "model_class_provider": "anthropic",
                }
            },
            "unit_primitives": [GitLabUnitPrimitive.DUO_CHAT],
            "prompt_template": {
                "system": "You are a helpful assistant.",
                "user": "{{user_input}}",
            },
            "params": {
                "stop": ["Human:", "Assistant:"],
                "timeout": 30.0,
            },
        }

        in_memory_config = InMemoryPromptConfig(**config_data)
        prompt_config = in_memory_config.to_prompt_config()

        # Should be a PromptConfig instance
        assert isinstance(prompt_config, PromptConfig)
        assert not isinstance(prompt_config, InMemoryPromptConfig)

        # Should have all fields except prompt_id
        assert prompt_config.name == "test_prompt"
        assert prompt_config.model.params.model == "claude-3-sonnet"
        assert prompt_config.unit_primitives == [GitLabUnitPrimitive.DUO_CHAT]
        assert prompt_config.prompt_template == {
            "system": "You are a helpful assistant.",
            "user": "{{user_input}}",
        }
        assert prompt_config.params.stop == ["Human:", "Assistant:"]
        assert prompt_config.params.timeout == 30.0

        # Should not have prompt_id field
        assert not hasattr(prompt_config, "prompt_id")
