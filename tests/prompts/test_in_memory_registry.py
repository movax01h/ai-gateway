from unittest.mock import Mock

import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive
from structlog.testing import capture_logs

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection.model_selection_config import (
    ChatLiteLLMDefinition,
    PromptParams,
)
from ai_gateway.model_selection.models import (
    BaseModelParams,
    ChatLiteLLMParams,
    OpenAIProviderParams,
    OpenAIReasoningParams,
)
from ai_gateway.prompts.base import Prompt, TemplateNotFoundError
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.config.base import (
    ModelConfig,
    PromptConfig,
    PromptProviderParams,
)
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
                    "model": "claude-sonnet-4-5-20250929",
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

    @staticmethod
    def _inline_prompt_data(extra_headers):
        """Build inline prompt data carrying the given model headers."""
        return {
            "model": {
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "model": "vertex_ai/gemini-2.5-flash",
                    "max_tokens": 1,
                    "max_retries": 0,
                    "extra_headers": extra_headers,
                },
            },
            "prompt_template": {"user": "say hello"},
            "unit_primitives": ["duo_agent_platform"],
        }

    @pytest.mark.parametrize(
        "extra_headers",
        [
            {"Host": "example.test"},
            {"X-Custom-Header": "value"},
            {"X-Goog-User-Project": "example"},
        ],
    )
    def test_inline_non_allowlisted_header_is_rejected(
        self, in_memory_registry, mock_shared_registry, extra_headers
    ):
        """Client-supplied inline model headers are restricted to the allowlist.

        When no trusted model metadata is attached, inline model params are used as-is; only allowlisted headers
        survive, and anything else fails the build before any model is created.
        """
        prompt_id = "duo_capture_prompt"
        in_memory_registry.register_prompt(
            prompt_id, self._inline_prompt_data(extra_headers)
        )

        with pytest.raises(ValueError, match="is not permitted"):
            in_memory_registry.get(prompt_id, prompt_version=None)

        mock_shared_registry._build_prompt.assert_not_called()

    def test_inline_non_allowlisted_default_headers_is_rejected(
        self, in_memory_registry, mock_shared_registry
    ):
        """Inline default_headers are validated too, not only extra_headers."""
        prompt_id = "duo_capture_prompt"
        in_memory_registry.register_prompt(
            prompt_id,
            {
                "model": {
                    "params": {
                        "model_class_provider": ModelClassProvider.LITE_LLM,
                        "model": "vertex_ai/gemini-2.5-flash",
                        "max_tokens": 1,
                        "default_headers": {"Host": "example.test"},
                    },
                },
                "prompt_template": {"user": "say hello"},
                "unit_primitives": ["duo_agent_platform"],
            },
        )

        with pytest.raises(ValueError, match="is not permitted"):
            in_memory_registry.get(prompt_id, prompt_version=None)

        mock_shared_registry._build_prompt.assert_not_called()

    def test_inline_allowlisted_header_is_permitted(
        self, in_memory_registry, mock_shared_registry, prompt
    ):
        """An allowlisted header still passes through the inline path."""
        prompt_id = "duo_capture_prompt"
        in_memory_registry.register_prompt(
            prompt_id, self._inline_prompt_data({"anthropic-beta": "feature-x"})
        )

        result = in_memory_registry.get(prompt_id, prompt_version=None)

        assert result == prompt
        mock_shared_registry._build_prompt.assert_called_once()

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

    @pytest.mark.parametrize(
        ("inline_params", "expected_keys", "expected_timeout"),
        [
            # No params in the flow config: the log still signals inline resolution.
            (None, None, None),
            # Params without a timeout.
            ({"stop": ["\n"]}, ["stop"], None),
            # A bound timeout in the client-sent flow config (e.g. injected by
            # Rails' DuoWorkflowPayloadBuilder) — the value that outranks
            # AIGW_DUO_CHAT__MODEL_REQUEST_TIMEOUT.
            ({"timeout": 30.0}, ["timeout"], 30.0),
            # The flow-config serializer normalizes params to the full
            # PromptParams schema with nulls; only keys actually set are
            # reported.
            (
                {
                    "cache_control_injection_points": None,
                    "context_management": None,
                    "model_id": None,
                    "stop": None,
                    "timeout": 360.0,
                    "vertex_location": None,
                },
                ["timeout"],
                360.0,
            ),
        ],
    )
    def test_get_local_prompt_logs_inline_resolution(
        self,
        in_memory_registry,
        sample_prompt_data,
        inline_params,
        expected_keys,
        expected_timeout,
    ):
        """Resolving an inline prompt emits a structured record with the params the flow config carried."""
        prompt_id = "test_prompt"
        prompt_data = {**sample_prompt_data}
        if inline_params is not None:
            prompt_data["params"] = inline_params
        in_memory_registry.register_prompt(prompt_id, prompt_data)

        with capture_logs() as cap_logs:
            in_memory_registry.get(prompt_id, prompt_version=None)

        entry = next(
            log_entry
            for log_entry in cap_logs
            if log_entry["event"] == "Resolving inline flow prompt"
        )
        assert entry["prompt_id"] == prompt_id
        assert entry["inline_params_keys"] == expected_keys
        assert entry["inline_timeout"] == expected_timeout
        assert entry["has_model_metadata"] is False

    def test_get_versioned_prompt_does_not_log_inline_resolution(
        self, in_memory_registry, sample_prompt_data
    ):
        """Delegation to the file-based registry must not emit the inline-resolution record."""
        prompt_id = "test_prompt"
        in_memory_registry.register_prompt(prompt_id, sample_prompt_data)

        with capture_logs() as cap_logs:
            in_memory_registry.get(prompt_id, prompt_version="^1.0.0")

        events = [log_entry["event"] for log_entry in cap_logs]
        assert "Resolving inline flow prompt" not in events

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

    def test_provider_params_forwarded_from_flow_model(
        self, in_memory_registry, mock_shared_registry, sample_prompt_data
    ):
        prompt_id = "test_prompt"
        prompt_data = {
            **sample_prompt_data,
            "model": {
                **sample_prompt_data["model"],
                "provider_params": {
                    "openai": {
                        "verbosity": "low",
                        "reasoning": {"summary": "auto", "effort": 8},
                    }
                },
            },
        }

        in_memory_registry.register_prompt(prompt_id, prompt_data)
        in_memory_registry.get(prompt_id, prompt_version=None)

        config = mock_shared_registry._build_prompt.call_args.kwargs["config"]
        assert config.model.provider_params == PromptProviderParams(
            openai=OpenAIProviderParams(
                verbosity="low",
                reasoning=OpenAIReasoningParams(summary="auto", effort=8),
            )
        )

    def test_provider_params_forwarded_when_model_metadata_present(
        self, in_memory_registry, mock_shared_registry, sample_prompt_data
    ):
        # model_metadata makes the flow's params ignored, but provider_params
        # are provider-conditional and must still be forwarded.
        prompt_id = "test_prompt"
        prompt_data = {
            **sample_prompt_data,
            "model": {
                **sample_prompt_data["model"],
                "provider_params": {"openai": {"verbosity": "low"}},
            },
        }
        model_metadata = ModelMetadata(
            name="test",
            provider="test",
            llm_definition=ChatLiteLLMDefinition(
                gitlab_identifier="claude",
                name="claude",
                max_context_tokens=200000,
                params=ChatLiteLLMParams(model="claude-sonnet-4-5-20250929"),
            ),
        )

        in_memory_registry.register_prompt(prompt_id, prompt_data)
        in_memory_registry.get(
            prompt_id, prompt_version=None, model_metadata=model_metadata
        )

        config = mock_shared_registry._build_prompt.call_args.kwargs["config"]
        assert config.model.params == BaseModelParams()
        assert config.model.provider_params == PromptProviderParams(
            openai=OpenAIProviderParams(verbosity="low")
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

    @pytest.mark.parametrize("is_graph_node", [True, False])
    def test_get_versioned_prompt_forwards_is_graph_node(
        self, in_memory_registry, mock_shared_registry, prompt, is_graph_node
    ):
        """is_graph_node must be forwarded to shared_registry.get() for versioned prompts.

        Regression test: when is_graph_node=True is passed for a flow-node prompt
        that has no entry in unit_primitives.yml, InMemoryPromptRegistry must relay
        the flag so that LocalPromptRegistry._default_model_metadata can fall back to
        duo_agent_platform instead of raising "Invalid feature setting: <prompt_id>".
        """
        mock_shared_registry.get.return_value = prompt

        in_memory_registry.get(
            "secret_vulnerability_source_file_agent_prompt",
            prompt_version="1.0.0",
            is_graph_node=is_graph_node,
        )

        mock_shared_registry.get.assert_called_once()
        _, kwargs = mock_shared_registry.get.call_args
        assert kwargs["is_graph_node"] is is_graph_node

    def test_get_local_prompt_missing_prompt_template(self, in_memory_registry):
        """Test error when prompt_template key is missing."""
        prompt_id = "missing_template_prompt"
        invalid_prompt_data = {
            "model": {
                "params": {
                    "model_class_provider": ModelClassProvider.LITE_LLM,
                    "model": "claude-sonnet-4-5-20250929",
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
                            model="claude-sonnet-4-5-20250929",
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
                            model="claude-sonnet-4-5-20250929",
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

    @pytest.mark.parametrize(
        "requires_single_system_message, expected_system",
        [
            (True, "Static part.\nDynamic part."),
            (False, ["Static part.", "Dynamic part."]),
        ],
    )
    def test_system_messages_collapsed_when_model_requires_it(
        self,
        in_memory_registry,
        mock_shared_registry,
        prompt,
        requires_single_system_message,
        expected_system,
    ):
        model_metadata = ModelMetadata(
            name="test",
            provider="test",
            llm_definition=ChatLiteLLMDefinition(
                gitlab_identifier="qwen",
                name="Qwen",
                max_context_tokens=262144,
                requires_single_system_message=requires_single_system_message,
                params=ChatLiteLLMParams(model="qwen3"),
            ),
        )
        prompt_data = {
            "prompt_template": {
                "system": ["Static part.", "Dynamic part."],
                "user": "{{goal}}",
            },
            "unit_primitives": ["duo_chat"],
        }
        in_memory_registry.register_prompt("chat_agent_prompt", prompt_data)
        in_memory_registry.get(
            "chat_agent_prompt", prompt_version=None, model_metadata=model_metadata
        )

        call_kwargs = mock_shared_registry._build_prompt.call_args.kwargs
        assert call_kwargs["config"].prompt_template["system"] == expected_system


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

    def test_list_system_extracts_variables_from_all_items(self, in_memory_registry):
        """Variables are collected from every element in a list-valued system field."""
        in_memory_registry.register_prompt(
            "list_prompt",
            {
                "prompt_template": {
                    "system": [
                        "Static: {{ static_var }}",
                        "Dynamic: {{ dynamic_var }}",
                    ]
                }
            },
        )
        result = in_memory_registry.get_required_variables(
            "list_prompt", prompt_version=None
        )
        assert result == {"static_var", "dynamic_var"}

    def test_list_system_with_no_variables_returns_empty_set(self, in_memory_registry):
        """A list-valued system field with no Jinja2 variables returns an empty set."""
        in_memory_registry.register_prompt(
            "static_list_prompt",
            {
                "prompt_template": {
                    "system": ["No variables here", "Or here either"],
                }
            },
        )
        result = in_memory_registry.get_required_variables(
            "static_list_prompt", prompt_version=None
        )
        assert result == set()

    def test_list_with_duplicate_variables_deduplicated(self, in_memory_registry):
        """Variables appearing in multiple list items are deduplicated in the result."""
        in_memory_registry.register_prompt(
            "dup_list_prompt",
            {
                "prompt_template": {
                    "system": ["Hello {{ name }}", "Goodbye {{ name }}"],
                }
            },
        )
        result = in_memory_registry.get_required_variables(
            "dup_list_prompt", prompt_version=None
        )
        assert result == {"name"}


class TestJoinSystemMessages:
    def test_list_is_joined_with_newline(self):
        template = {"system": ["Static part.", "Dynamic part."], "user": "Hello"}
        result = InMemoryPromptRegistry._join_system_messages(template)
        assert result["system"] == "Static part.\nDynamic part."
        assert result["user"] == "Hello"

    def test_single_string_is_passed_through_unchanged(self):
        template: dict[str, str | list[str]] = {
            "system": "Already a string.",
            "user": "Hello",
        }
        result = InMemoryPromptRegistry._join_system_messages(template)
        assert result["system"] == "Already a string."

    def test_empty_string_is_passed_through_unchanged(self):
        template: dict[str, str | list[str]] = {"system": "", "user": "Hello"}
        result = InMemoryPromptRegistry._join_system_messages(template)
        assert result["system"] == ""

    def test_no_system_key_returns_template_unchanged(self):
        template: dict[str, str | list[str]] = {"user": "Hello"}
        result = InMemoryPromptRegistry._join_system_messages(template)
        assert "system" not in result
        assert result["user"] == "Hello"
