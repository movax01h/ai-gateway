from unittest.mock import Mock

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection.model_selection_config import (
    ChatLiteLLMDefinition,
    PromptParams,
)
from ai_gateway.model_selection.models import BaseModelParams, ChatLiteLLMParams
from ai_gateway.prompts.base import Prompt, TemplateNotFoundError
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.in_memory_registry import InMemoryPromptRegistry
from ai_gateway.prompts.registry import LocalPromptRegistry
from lib.internal_events.client import InternalEventsClient


class TestInMemoryPromptRegistry:

    @pytest.fixture
    def mock_shared_registry(
        self,
        prompt: Prompt,
        internal_event_client: InternalEventsClient,
        model_limits: ConfigModelLimits,
    ):
        """Mock the shared LocalPromptRegistry."""
        registry = Mock(spec=LocalPromptRegistry)
        registry.internal_event_client = internal_event_client
        registry.model_limits = model_limits
        registry._build_prompt.return_value = prompt
        registry.get.return_value = prompt
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

    def test_get_local_prompt_success(
        self, in_memory_registry, mock_shared_registry, sample_prompt_data, prompt
    ):
        """Test successful retrieval of local prompt with prompt_version=None."""
        prompt_id = "test_prompt"

        # Setup: register prompt
        in_memory_registry.register_prompt(prompt_id, sample_prompt_data)

        # Test: get with prompt_version=None
        result = in_memory_registry.get(prompt_id, prompt_version=None)

        mock_shared_registry._build_prompt.assert_called_once()
        assert result == prompt

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
        self,
        in_memory_registry,
        prompt,
        sample_prompt_data,
        prompt_version,
        should_use_shared,
    ):
        """Test routing logic with various prompt versions."""
        prompt_id = "test_prompt"

        # Register local prompt
        in_memory_registry.register_prompt(prompt_id, sample_prompt_data)

        # Test routing
        result = in_memory_registry.get(prompt_id, prompt_version)

        if should_use_shared:
            # Should use shared registry
            in_memory_registry.shared_registry.get.assert_called_once()
        else:
            in_memory_registry.shared_registry._build_prompt.assert_called_once()

        assert result == prompt

    @pytest.mark.parametrize(
        "unit_primitives,expected_unit_primitive",
        [
            (None, GitLabUnitPrimitive.DUO_AGENT_PLATFORM),
            ([], GitLabUnitPrimitive.DUO_AGENT_PLATFORM),
            (["duo_chat"], GitLabUnitPrimitive.DUO_CHAT),
        ],
    )
    def test_prompt_config_conversion(
        self,
        in_memory_registry,
        mock_shared_registry,
        sample_prompt_data,
        unit_primitives,
        expected_unit_primitive,
    ):
        """Test that flow YAML data is correctly converted to PromptConfig."""
        prompt_id = "test_prompt"

        # Add optional fields to test defaults
        extended_data = {
            **sample_prompt_data,
            "unit_primitives": unit_primitives,
            "params": {"timeout": 30},
        }

        in_memory_registry.register_prompt(prompt_id, extended_data)

        in_memory_registry.get(prompt_id, prompt_version=None)

        mock_shared_registry._build_prompt.assert_called_once_with(
            model_class_provider=ModelClassProvider.LITE_LLM,
            config=PromptConfig(
                name=prompt_id,
                model=ModelConfig(params=sample_prompt_data["model"]["params"]),
                unit_primitive=expected_unit_primitive,
                prompt_template=sample_prompt_data["prompt_template"],
                params=PromptParams(timeout=30.0),
            ),
            model_metadata=None,
            tool_choice=None,
            tools=None,
        )

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

        with pytest.raises(KeyError, match="'prompt_template'"):
            in_memory_registry.get(prompt_id, prompt_version=None)

    @pytest.mark.parametrize(
        "raw_model_data,model_metadata,expected_model_params,expected_model_class_provider",
        [
            (
                {
                    "params": {
                        "model": "claude-3-7-sonnet-20250219",
                        "model_class_provider": ModelClassProvider.LITE_LLM,
                    }
                },
                None,
                BaseModelParams(model="claude-3-7-sonnet-20250219"),
                ModelClassProvider.LITE_LLM,
            ),
            (
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
                BaseModelParams(),
                ModelClassProvider.LITE_LLM,
            ),
            (
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
                BaseModelParams(),
                ModelClassProvider.LITE_LLM,
            ),
        ],
    )
    def test_model_class_provider_and_data_handling(
        self,
        in_memory_registry,
        mock_shared_registry,
        raw_model_data,
        model_metadata,
        expected_model_params,
        expected_model_class_provider,
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

        in_memory_registry.get(
            prompt_id, prompt_version=None, model_metadata=model_metadata
        )

        # Verify the model_class_provider passed to _build_prompt
        mock_shared_registry._build_prompt.assert_called_once()
        call_kwargs = mock_shared_registry._build_prompt.call_args.kwargs
        assert call_kwargs["model_class_provider"] == expected_model_class_provider
        prompt_config = call_kwargs["config"]
        assert prompt_config.model.params == expected_model_params


class TestGetRequiredVariables:

    @pytest.fixture
    def mock_shared_registry(self, internal_event_client, model_limits):
        registry = Mock(spec=LocalPromptRegistry)
        registry.internal_event_client = internal_event_client
        registry.model_limits = model_limits
        return registry

    @pytest.fixture
    def in_memory_registry(self, mock_shared_registry):
        return InMemoryPromptRegistry(mock_shared_registry)

    def test_inline_prompt_returns_variables(self, in_memory_registry):
        in_memory_registry.register_prompt(
            "my_prompt",
            {
                "prompt_template": {
                    "system": "Hello {{ name }}, your goal is {{ goal }}"
                }
            },
        )
        result = in_memory_registry.get_required_variables(
            "my_prompt", prompt_version=None
        )
        assert result == {"name", "goal"}

    def test_inline_prompt_not_found_raises(self, in_memory_registry):
        with pytest.raises(TemplateNotFoundError):
            in_memory_registry.get_required_variables("missing", prompt_version=None)

    def test_versioned_prompt_delegates_to_shared_registry(
        self, in_memory_registry, mock_shared_registry
    ):
        mock_shared_registry.get_required_variables.return_value = {"foo"}
        result = in_memory_registry.get_required_variables(
            "my_prompt", prompt_version="^1.0.0"
        )
        assert result == {"foo"}
        mock_shared_registry.get_required_variables.assert_called_once_with(
            "my_prompt", "^1.0.0"
        )
