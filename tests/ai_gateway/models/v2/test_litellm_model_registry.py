"""Tests for the external LiteLLM model metadata registry."""

import json
from pathlib import Path
from unittest.mock import patch

import litellm
import pytest
from litellm.utils import supports_tool_choice

from ai_gateway.model_selection.model_selection_config import (
    ChatAnthropicDefinition,
    ChatLiteLLMDefinition,
    CompletionLiteLLMDefinition,
    EmbeddingLiteLLMDefinition,
)
from ai_gateway.model_selection.models import (
    ChatAnthropicParams,
    ChatLiteLLMParams,
    CompletionLiteLLMParams,
    CompletionType,
    EmbeddingLiteLLMParams,
)
from ai_gateway.models.v2.litellm_model_registry import (
    BUILTIN_MODEL_METADATA,
    ENV_VAR_NAME,
    load_external_model_metadata,
    register_builtin_models,
    register_external_models,
    register_fireworks_models,
)

VALID_JSON = json.dumps(
    {
        "models": {
            "fireworks_ai/accounts/gitlab/deployments/test-model": {
                "litellm_provider": "fireworks_ai",
                "mode": "chat",
                "max_input_tokens": 262144,
                "max_output_tokens": 262144,
                "supports_function_calling": True,
                "supports_tool_choice": True,
                "supports_response_schema": True,
            },
            "hosted_vllm/another-model": {
                "litellm_provider": "openai",
                "mode": "chat",
                "max_input_tokens": 131072,
                "supports_function_calling": True,
                "supports_tool_choice": True,
            },
        }
    }
)

REGISTER_MODEL = "ai_gateway.models.v2.litellm_model_registry.register_model"


@pytest.fixture(name="mock_register")
def mock_register_fixture():
    """Patch ``register_model`` so tests never mutate LiteLLM's real registry."""
    with patch(REGISTER_MODEL) as mock:
        yield mock


class TestRegisterBuiltinModels:
    """Tests for ``register_builtin_models``."""

    def test_success_registers_models(self, mock_register) -> None:
        """When ``register_model`` succeeds, built-in models are registered."""
        register_builtin_models()

        mock_register.assert_called_once()

    def test_exception_does_not_propagate(self, mock_register) -> None:
        """If ``register_model`` raises, the function returns without re-raising."""
        mock_register.side_effect = RuntimeError("litellm internal error")

        # Should not raise
        register_builtin_models()


class TestSonnet5BedrockMetadata:
    """Tests for the manually-registered Claude Sonnet 5 Bedrock metadata."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "global.anthropic.claude-sonnet-5-v1:0",
            "us.anthropic.claude-sonnet-5-v1:0",
            "eu.anthropic.claude-sonnet-5-v1:0",
            "bedrock/global.anthropic.claude-sonnet-5",
        ],
    )
    def test_builtin_metadata_has_sonnet_5_bedrock_keys(self, model_name: str) -> None:
        """Every Sonnet 5 Bedrock cross-region inference profile is registered."""
        assert model_name in BUILTIN_MODEL_METADATA
        assert (
            BUILTIN_MODEL_METADATA[model_name]["litellm_provider"] == "bedrock_converse"
        )
        assert BUILTIN_MODEL_METADATA[model_name]["supports_tool_choice"] is True

    def test_registered_with_litellm(self, mock_register) -> None:
        """register_builtin_models passes the Sonnet 5 Bedrock model string used in models.yml to LiteLLM."""
        register_builtin_models()

        registered = mock_register.call_args.args[0]
        model_name = "bedrock/global.anthropic.claude-sonnet-5"
        assert registered[model_name]["supports_tool_choice"] is True
        assert registered[model_name]["max_output_tokens"] == 64_000


class TestLoadExternalModelMetadata:
    """Tests for ``load_external_model_metadata``."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """A valid JSON file is parsed into the expected dict."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)

        result = load_external_model_metadata(str(file_path))

        assert len(result) == 2
        assert "fireworks_ai/accounts/gitlab/deployments/test-model" in result
        assert "hosted_vllm/another-model" in result
        fireworks_meta = result["fireworks_ai/accounts/gitlab/deployments/test-model"]
        assert fireworks_meta["supports_tool_choice"] is True
        assert fireworks_meta["litellm_provider"] == "fireworks_ai"
        assert fireworks_meta["max_input_tokens"] == 262144

    def test_missing_file_raises_filenotfounderror(self, tmp_path: Path) -> None:
        """A non-existent path raises ``FileNotFoundError``."""
        missing = tmp_path / "does-not-exist.json"

        with pytest.raises(FileNotFoundError):
            load_external_model_metadata(str(missing))

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """An empty file yields an empty dict (no models to register)."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("")

        result = load_external_model_metadata(str(file_path))

        assert result == {}

    def test_json_without_models_key_returns_empty_dict(self, tmp_path: Path) -> None:
        """A JSON file lacking the ``models`` key yields an empty dict."""
        file_path = tmp_path / "no_models.json"
        file_path.write_text(json.dumps({"other_key": "value"}))

        result = load_external_model_metadata(str(file_path))

        assert result == {}

    def test_invalid_json_raises_jsondecodeerror(self, tmp_path: Path) -> None:
        """Malformed JSON raises ``json.JSONDecodeError``."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text('{"models": {"foo": [unclosed')

        with pytest.raises(json.JSONDecodeError):
            load_external_model_metadata(str(file_path))

    def test_top_level_not_object_raises_valueerror(self, tmp_path: Path) -> None:
        """A JSON file whose top level is not an object raises ``ValueError``."""
        file_path = tmp_path / "list.json"
        file_path.write_text(json.dumps(["item1", "item2"]))

        with pytest.raises(ValueError, match="top level"):
            load_external_model_metadata(str(file_path))

    def test_models_not_object_raises_valueerror(self, tmp_path: Path) -> None:
        """A non-object ``models`` value raises ``ValueError``."""
        file_path = tmp_path / "bad_models.json"
        file_path.write_text(json.dumps({"models": ["not-a-mapping"]}))

        with pytest.raises(ValueError, match="`models` to be a JSON object"):
            load_external_model_metadata(str(file_path))

    def test_model_entry_not_object_raises_valueerror(self, tmp_path: Path) -> None:
        """An individual model entry that is not an object raises ``ValueError``."""
        file_path = tmp_path / "bad_entry.json"
        file_path.write_text(json.dumps({"models": {"my-model": "not-a-mapping"}}))

        with pytest.raises(ValueError, match="model `my-model`"):
            load_external_model_metadata(str(file_path))


class TestRegisterExternalModels:
    """Tests for ``register_external_models``."""

    def test_no_env_var_is_noop(
        self, mock_register, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the env var is unset, ``register_model`` is not called."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        register_external_models()

        mock_register.assert_not_called()

    def test_empty_env_var_is_noop(self, mock_register) -> None:
        """An empty string env var is treated as unset."""
        with patch.dict("os.environ", {ENV_VAR_NAME: ""}):
            register_external_models()

        mock_register.assert_not_called()

    def test_explicit_file_path_registers_models(
        self, mock_register, tmp_path: Path
    ) -> None:
        """An explicit ``file_path`` argument loads and registers models."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)

        register_external_models(file_path=str(file_path))

        mock_register.assert_called_once()
        registered = mock_register.call_args.args[0]
        assert len(registered) == 2
        assert "fireworks_ai/accounts/gitlab/deployments/test-model" in registered

    def test_env_var_file_path_registers_models(
        self, mock_register, tmp_path: Path
    ) -> None:
        """The env var is honored when no explicit path is given."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)

        with patch.dict("os.environ", {ENV_VAR_NAME: str(file_path)}):
            register_external_models()

        mock_register.assert_called_once()

    def test_missing_file_logs_warning_and_continues(
        self, mock_register, tmp_path: Path
    ) -> None:
        """Missing files do not raise; ``register_model`` is not called."""
        missing = tmp_path / "missing.json"

        # Should not raise
        register_external_models(file_path=str(missing))

        mock_register.assert_not_called()

    def test_invalid_json_logs_warning_and_continues(
        self, mock_register, tmp_path: Path
    ) -> None:
        """Invalid JSON does not raise; ``register_model`` is not called."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text('{"models": {"foo": [unclosed')

        register_external_models(file_path=str(file_path))

        mock_register.assert_not_called()

    def test_invalid_structure_logs_warning_and_continues(
        self, mock_register, tmp_path: Path
    ) -> None:
        """Structural validation errors do not raise; registration is skipped."""
        file_path = tmp_path / "bad_structure.json"
        file_path.write_text(json.dumps({"models": {"my-model": "not-a-mapping"}}))

        register_external_models(file_path=str(file_path))

        mock_register.assert_not_called()

    def test_empty_models_section_does_not_call_register(
        self, mock_register, tmp_path: Path
    ) -> None:
        """An empty ``models`` section results in no registration call."""
        file_path = tmp_path / "empty.json"
        file_path.write_text(json.dumps({"models": {}}))

        register_external_models(file_path=str(file_path))

        mock_register.assert_not_called()

    def test_register_model_exception_does_not_propagate(
        self, mock_register, tmp_path: Path
    ) -> None:
        """If ``register_model`` raises, the application still starts."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)
        mock_register.side_effect = RuntimeError("litellm internal error")

        # Should not raise
        register_external_models(file_path=str(file_path))


class TestIntegrationWithLiteLLM:
    """Integration tests verifying real LiteLLM registration."""

    @pytest.fixture(autouse=True)
    def _restore_model_cost(self):
        """Roll back any real registrations so the global registry stays clean."""
        with patch.dict(litellm.model_cost, {}, clear=False):
            yield

    def test_registered_model_is_recognized_by_litellm(self, tmp_path: Path) -> None:
        """A model registered via external metadata is recognized by LiteLLM."""
        model_name = "fireworks_ai/accounts/gitlab/deployments/test-aigw-tool-choice"
        file_path = tmp_path / "models.json"
        file_path.write_text(
            json.dumps(
                {
                    "models": {
                        model_name: {
                            "litellm_provider": "fireworks_ai",
                            "mode": "chat",
                            "max_input_tokens": 262144,
                            "max_output_tokens": 262144,
                            "supports_function_calling": True,
                            "supports_tool_choice": True,
                        }
                    }
                }
            )
        )

        register_external_models(file_path=str(file_path))

        # Verify the model is now in LiteLLM's registry
        assert model_name in litellm.model_cost
        assert litellm.model_cost[model_name].get("supports_tool_choice") is True

    def test_fireworks_model_from_config_supports_tool_choice(self) -> None:
        """A Fireworks LLM definition registered via register_fireworks_models is recognized by LiteLLM."""
        definition = ChatLiteLLMDefinition(
            name="Integration Test",
            gitlab_identifier="integration_test_fireworks",
            max_context_tokens=200_000,
            cost_indicator="$",
            description="Integration test Fireworks model.",
            params=ChatLiteLLMParams(
                model="accounts/gitlab/deployments/test-aigw-2587",
                custom_llm_provider="fireworks_ai",
                max_tokens=8_192,
            ),
        )

        register_fireworks_models({"m": definition})

        key = "fireworks_ai/accounts/gitlab/deployments/test-aigw-2587"
        assert key in litellm.model_cost
        assert supports_tool_choice(key) is True


class TestRegisterFireworksModels:
    """Tests for ``register_fireworks_models``."""

    @pytest.fixture(name="fireworks_chat_definition")
    def fireworks_chat_definition_fixture(self) -> ChatLiteLLMDefinition:
        return ChatLiteLLMDefinition(
            name="GLM Test",
            gitlab_identifier="glm_test_fireworks",
            max_context_tokens=200_000,
            cost_indicator="$",
            description="Test Fireworks chat model.",
            params=ChatLiteLLMParams(
                model="accounts/gitlab/deployments/test123",
                custom_llm_provider="fireworks_ai",
                max_tokens=8_192,
            ),
        )

    def test_registers_chat_model_with_capability_flags(
        self, mock_register, fireworks_chat_definition: ChatLiteLLMDefinition
    ) -> None:
        """A Fireworks chat model is registered with tool-calling capability flags."""
        register_fireworks_models({"glm": fireworks_chat_definition})

        registered = mock_register.call_args.args[0]
        entry = registered["fireworks_ai/accounts/gitlab/deployments/test123"]
        assert entry["litellm_provider"] == "fireworks_ai"
        assert entry["mode"] == "chat"
        assert entry["supports_function_calling"] is True
        assert entry["supports_tool_choice"] is True
        assert entry["max_input_tokens"] == 200_000
        assert entry["max_output_tokens"] == 8_192

    def test_registers_both_model_and_identifier_keys(self, mock_register) -> None:
        """Router-style entries register both the model name and the identifier."""
        definition = ChatLiteLLMDefinition(
            name="Router Test",
            gitlab_identifier="router_test_fireworks",
            max_context_tokens=32_000,
            cost_indicator="$",
            description="Test Fireworks chat model with router identifier.",
            params=ChatLiteLLMParams(
                model="glm-test",
                custom_llm_provider="fireworks_ai",
                identifier="accounts/gitlab/routers/glm-test",
            ),
        )

        register_fireworks_models({"glm": definition})

        registered = mock_register.call_args.args[0]
        assert "fireworks_ai/glm-test" in registered
        assert "fireworks_ai/accounts/gitlab/routers/glm-test" in registered

    @pytest.mark.parametrize(
        "definition",
        [
            ChatAnthropicDefinition(
                name="Claude Test",
                gitlab_identifier="claude_test",
                max_context_tokens=200_000,
                cost_indicator="$$",
                description="Non-Fireworks model.",
                params=ChatAnthropicParams(model="claude-test-model"),
            ),
            EmbeddingLiteLLMDefinition(
                name="Embedding Test",
                gitlab_identifier="embedding_test_fireworks",
                max_context_tokens=8_192,
                cost_indicator="$",
                description="Fireworks embedding model.",
                params=EmbeddingLiteLLMParams(
                    model="test-embedding",
                    custom_llm_provider="fireworks_ai",
                ),
            ),
            CompletionLiteLLMDefinition(
                name="Codestral Test",
                gitlab_identifier="codestral_test_fireworks",
                max_context_tokens=32_000,
                cost_indicator="$",
                description="Fireworks FIM completion model.",
                params=CompletionLiteLLMParams(
                    model="codestral-test",
                    custom_llm_provider="fireworks_ai",
                    identifier="accounts/gitlab/routers/codestral-test",
                    completion_type=CompletionType.FIM,
                    fim_format="</s>[SUFFIX]{suffix}[PREFIX]{prefix}[MIDDLE]",
                ),
            ),
        ],
    )
    def test_skips_non_chat_and_non_fireworks_models(
        self, mock_register, definition
    ) -> None:
        """Non-Fireworks providers, embedding, and completion models are not registered."""
        register_fireworks_models({"model": definition})

        mock_register.assert_not_called()

    def test_skips_models_already_in_litellm_registry(
        self, mock_register, fireworks_chat_definition: ChatLiteLLMDefinition
    ) -> None:
        """Existing LiteLLM entries (bundled or external) are never clobbered."""
        existing_key = "fireworks_ai/accounts/gitlab/deployments/test123"
        with patch.dict(litellm.model_cost, {existing_key: {"mode": "chat"}}):
            register_fireworks_models({"glm": fireworks_chat_definition})

        mock_register.assert_not_called()

    def test_register_model_exception_does_not_propagate(
        self, mock_register, fireworks_chat_definition: ChatLiteLLMDefinition
    ) -> None:
        """If ``register_model`` raises, the application still starts."""
        mock_register.side_effect = RuntimeError("litellm internal error")

        # Should not raise
        register_fireworks_models({"glm": fireworks_chat_definition})
