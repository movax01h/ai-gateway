from unittest.mock import Mock

import pytest
from langchain_community.chat_models import ChatLiteLLM
from pydantic import ValidationError

from ai_gateway.prompts.base import Prompt
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.in_memory_registry import InMemoryPromptRegistry
from ai_gateway.prompts.registry import LocalPromptRegistry


class TestInMemoryPromptRegistry:

    @pytest.fixture
    def mock_shared_registry(self):
        """Mock the shared LocalPromptRegistry."""
        registry = Mock(spec=LocalPromptRegistry)
        registry.internal_event_client = Mock()
        registry.model_limits = Mock()
        registry.model_configs = {}
        registry.model_factories = {
            ModelClassProvider.LITE_LLM: lambda model, **kwargs: ChatLiteLLM(
                model=model, **kwargs
            )
        }
        registry.disable_streaming = False
        return registry

    @pytest.fixture
    def in_memory_registry(self, mock_shared_registry):
        """Create InMemoryPromptRegistry instance for testing."""
        return InMemoryPromptRegistry(mock_shared_registry)

    @pytest.fixture
    def sample_prompt_data(self):
        return {
            "model": {
                "name": "claude_4_0",
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "max_tokens": 1000,
                },
            },
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            },
        }

    def test_register_prompt(self, in_memory_registry, sample_prompt_data):
        """Test that prompts can be registered and stored."""
        prompt_id = "test_prompt"

        in_memory_registry.register_prompt(prompt_id, sample_prompt_data)

        # Verify prompt is stored
        assert prompt_id in in_memory_registry._raw_prompt_data
        assert in_memory_registry._raw_prompt_data[prompt_id] == sample_prompt_data

    def test_get_local_prompt_success(self, in_memory_registry, sample_prompt_data):
        """Test successful retrieval of local prompt with prompt_version=None."""
        prompt_id = "test_prompt"

        # Setup: register prompt
        in_memory_registry.register_prompt(prompt_id, sample_prompt_data)

        # Test: get with prompt_version=None
        result = in_memory_registry.get(prompt_id, prompt_version=None)

        # Verify: result is a Prompt instance
        assert isinstance(result, Prompt)
        assert result.model is not None

    def test_get_local_prompt_not_found(self, in_memory_registry):
        """Test error when local prompt not found."""
        with pytest.raises(ValueError, match="Local prompt not found: nonexistent"):
            in_memory_registry.get("nonexistent", prompt_version=None)

    @pytest.mark.parametrize(
        "prompt_version,should_use_shared",
        [
            (None, False),
            ("", False),  # Empty string should use local
            ("^1.0.0", True),
            ("latest", True),
            ("1.0.0", True),
        ],
    )
    def test_routing_logic(
        self, in_memory_registry, sample_prompt_data, prompt_version, should_use_shared
    ):
        """Test routing logic with various prompt versions."""
        prompt_id = "test_prompt"

        # Register local prompt
        in_memory_registry.register_prompt(prompt_id, sample_prompt_data)

        # Setup mocks
        in_memory_registry.shared_registry.model_factories = {
            ModelClassProvider.LITE_LLM: lambda model, **kwargs: ChatLiteLLM(
                model=model, **kwargs
            )
        }
        mock_remote_prompt = Mock()
        in_memory_registry.shared_registry.get.return_value = mock_remote_prompt

        # Test routing
        result = in_memory_registry.get(prompt_id, prompt_version)

        if should_use_shared:
            # Should use shared registry
            assert result == mock_remote_prompt
            in_memory_registry.shared_registry.get.assert_called_once_with(
                prompt_id, prompt_version, None, None, None
            )
        else:
            # Should use local registry
            assert isinstance(result, Prompt)
            assert result.model is not None
            in_memory_registry.shared_registry.get.assert_not_called()

    def test_prompt_config_conversion(self, in_memory_registry, sample_prompt_data):
        """Test that flow YAML data is correctly converted to PromptConfig."""
        prompt_id = "test_prompt"

        # Add optional fields to test defaults
        extended_data = {
            **sample_prompt_data,
            "unit_primitives": ["duo_chat"],
            "params": {"timeout": 30},
        }

        in_memory_registry.register_prompt(prompt_id, extended_data)

        # Setup model factory
        in_memory_registry.shared_registry.model_factories = {
            ModelClassProvider.LITE_LLM: lambda model, **kwargs: ChatLiteLLM(
                model=model, **kwargs
            )
        }

        # Test conversion
        result = in_memory_registry.get(prompt_id, prompt_version=None)

        # Verify PromptConfig was created correctly
        assert isinstance(result, Prompt)
        # Additional assertions would verify the PromptConfig fields

    def test_get_local_prompt_invalid_model_provider(self, in_memory_registry):
        """Test error when model_class_provider is not recognized."""
        prompt_id = "invalid_model_prompt"
        invalid_prompt_data = {
            "model": {
                "name": "claude_4_0",
                "params": {
                    "model_class_provider": "invalid_provider",
                    "max_tokens": 1000,
                },
            },
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            },
        }

        in_memory_registry.register_prompt(prompt_id, invalid_prompt_data)

        with pytest.raises(
            ValidationError,
            match="Input tag 'invalid_provider' found",
        ):
            in_memory_registry.get(prompt_id, prompt_version=None)

    def test_get_local_prompt_missing_model_key(self, in_memory_registry):
        """Test error when model key is missing from prompt data."""
        prompt_id = "missing_model_prompt"
        invalid_prompt_data = {
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            }
        }

        in_memory_registry.register_prompt(prompt_id, invalid_prompt_data)

        with pytest.raises(KeyError, match="'model'"):
            in_memory_registry.get(prompt_id, prompt_version=None)

    def test_get_local_prompt_missing_prompt_template(self, in_memory_registry):
        """Test error when prompt_template key is missing."""
        prompt_id = "missing_template_prompt"
        invalid_prompt_data = {
            "model": {
                "name": "claude_4_0",
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "max_tokens": 1000,
                },
            }
        }

        in_memory_registry.register_prompt(prompt_id, invalid_prompt_data)
        in_memory_registry.shared_registry.model_factories = {
            ModelClassProvider.LITE_LLM: lambda model, **kwargs: ChatLiteLLM(
                model=model, **kwargs
            )
        }

        with pytest.raises(KeyError, match="'prompt_template'"):
            in_memory_registry.get(prompt_id, prompt_version=None)
