from unittest.mock import Mock, patch

import pytest
from langchain_community.chat_models import ChatAnthropic

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection.model_selection_config import ChatLiteLLMDefinition
from ai_gateway.model_selection.models import ChatLiteLLMParams
from ai_gateway.prompts.base import Prompt
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.in_memory_registry import InMemoryPromptRegistry
from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM


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
            ),
            ModelClassProvider.ANTHROPIC: lambda model, **kwargs: ChatAnthropic(
                model=model, **kwargs
            ),
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
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "model": "claude-sonnet-4-20250514",
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
                "params": {
                    "model_class_provider": "invalid_provider",
                    "model": "claude-sonnet-4-20250514",
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
            ValueError,
            match="unrecognized model class provider `invalid_provider`",
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

        with pytest.raises(
            ValueError, match=f"Model config not provided for prompt {prompt_id}"
        ):
            in_memory_registry.get(prompt_id, prompt_version=None)

    def test_get_local_prompt_missing_prompt_template(self, in_memory_registry):
        """Test error when prompt_template key is missing."""
        prompt_id = "missing_template_prompt"
        invalid_prompt_data = {
            "model": {
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "model": "claude-sonnet-4-20250514",
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

    def test_model_class_provider_from_metadata_overrides_yaml(
        self, in_memory_registry
    ):
        """When model_metadata provides model_class_provider, it overrides the YAML config."""
        prompt_id = "test_prompt"
        prompt_data = {
            "model": {
                "params": {
                    "model_class_provider": ModelClassProvider.ANTHROPIC,
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                },
            },
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            },
        }
        in_memory_registry.register_prompt(prompt_id, prompt_data)

        # model_metadata says litellm, YAML says anthropic — metadata should win
        metadata = ModelMetadata(
            name="self-hosted",
            provider="custom",
            llm_definition=ChatLiteLLMDefinition(
                gitlab_identifier="litellm_proxy",
                name="litellm_proxy",
                max_context_tokens=200000,
                params=ChatLiteLLMParams(
                    model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0"
                ),
            ),
        )

        result = in_memory_registry.get(
            prompt_id, prompt_version=None, model_metadata=metadata
        )

        assert isinstance(result, Prompt)
        assert isinstance(result.model, ChatLiteLLM)

    def test_model_class_provider_falls_back_to_yaml_when_no_metadata(
        self, in_memory_registry
    ):
        """When model_metadata is None, model_class_provider comes from YAML (anthropic factory from config)."""
        prompt_id = "test_prompt"
        prompt_data = {
            "model": {
                "params": {
                    "model_class_provider": ModelClassProvider.ANTHROPIC,
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                },
            },
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            },
        }
        in_memory_registry.register_prompt(prompt_id, prompt_data)

        with patch("ai_gateway.prompts.in_memory_registry.Prompt") as prompt_class:
            in_memory_registry.get(prompt_id, prompt_version=None, model_metadata=None)

        # The anthropic factory should have been selected (from YAML), not litellm
        call_kwargs = prompt_class.call_args.kwargs
        factory = call_kwargs.get("model_factory")
        assert (
            factory
            is in_memory_registry.shared_registry.model_factories[
                ModelClassProvider.ANTHROPIC
            ]
        )

    def test_tool_choice_adjusted_for_bedrock(self, in_memory_registry):
        """When model_metadata has a Bedrock identifier, tool_choice='any' becomes 'required'."""
        prompt_id = "test_prompt"
        prompt_data = {
            "model": {
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                },
            },
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            },
        }
        in_memory_registry.register_prompt(prompt_id, prompt_data)

        bedrock_metadata = ModelMetadata(
            name="bedrock_model",
            provider="custom",
            identifier="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            llm_definition=ChatLiteLLMDefinition(
                gitlab_identifier="bedrock_claude",
                name="bedrock_claude",
                max_context_tokens=200000,
                params=ChatLiteLLMParams(
                    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
                ),
            ),
        )

        # Wire in the real method so tool_choice adjustment actually runs
        in_memory_registry.shared_registry._adjust_tool_choice_for_model = (
            lambda tc, mm: LocalPromptRegistry._adjust_tool_choice_for_model(
                in_memory_registry.shared_registry, tc, mm
            )
        )

        with patch("ai_gateway.prompts.in_memory_registry.Prompt") as prompt_class:
            in_memory_registry.get(
                prompt_id,
                prompt_version=None,
                model_metadata=bedrock_metadata,
                tool_choice="any",
            )

        kwargs = prompt_class.call_args.kwargs
        assert kwargs.get("tool_choice") == "required"

    @pytest.mark.parametrize(
        "raw_model_data,model_metadata,expected_result",
        [
            (
                # Only the prompt model is provided
                {
                    "params": {
                        "model": "claude-3-7-sonnet-20250219",
                        "model_class_provider": ModelClassProvider.LITE_LLM,
                    }
                },
                None,
                "claude-3-7-sonnet-20250219",
            ),
            (
                # Only the model_metadata is provided
                None,
                ModelMetadata(
                    name="test",
                    provider="test",
                    llm_definition=ChatLiteLLMDefinition(
                        gitlab_identifier="claude",
                        name="claude",
                        max_context_tokens=200000,
                        params=ChatLiteLLMParams(
                            model="claude-sonnet-4-20250514",
                        ),
                    ),
                ),
                "claude-sonnet-4-20250514",
            ),
            (
                # Both the prompt model and the model_metadata are provided; we use the model_metadata
                {
                    "params": {
                        "model": "claude-3-7-sonnet-20250219",
                        "model_class_provider": ModelClassProvider.LITE_LLM,
                    }
                },
                ModelMetadata(
                    name="test",
                    provider="test",
                    llm_definition=ChatLiteLLMDefinition(
                        gitlab_identifier="claude",
                        name="claude",
                        max_context_tokens=200000,
                        params=ChatLiteLLMParams(
                            model="claude-sonnet-4-20250514",
                        ),
                    ),
                ),
                "claude-sonnet-4-20250514",
            ),
        ],
    )
    def test_model_data_handling(
        self,
        in_memory_registry,
        raw_model_data,
        model_metadata,
        expected_result,
    ):
        prompt_id = "test_prompt"
        prompt_data = {
            "prompt_template": {
                "system": "You are a helpful assistant",
                "user": "Task: {{goal}}",
            }
        }
        if raw_model_data:
            prompt_data["model"] = raw_model_data

        in_memory_registry.register_prompt(prompt_id, prompt_data)

        result = in_memory_registry.get(
            prompt_id, prompt_version=None, model_metadata=model_metadata
        )
        assert isinstance(result, Prompt)
        assert result.model.model == expected_result
