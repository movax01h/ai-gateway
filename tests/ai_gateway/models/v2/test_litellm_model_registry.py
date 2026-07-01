"""Tests for the external LiteLLM model metadata registry."""

import json
from pathlib import Path
from unittest.mock import patch

import litellm
import pytest

from ai_gateway.models.v2.litellm_model_registry import (
    BUILTIN_MODEL_METADATA,
    ENV_VAR_NAME,
    load_external_model_metadata,
    register_builtin_models,
    register_external_models,
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


class TestRegisterBuiltinModels:
    """Tests for ``register_builtin_models``."""

    def test_success_registers_models(self) -> None:
        """When ``register_model`` succeeds, built-in models are registered."""
        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            register_builtin_models()
            mock_register.assert_called_once()

    def test_exception_does_not_propagate(self) -> None:
        """If ``register_model`` raises, the function returns without re-raising."""
        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model",
            side_effect=RuntimeError("litellm internal error"),
        ):
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

    def test_registered_with_litellm(self) -> None:
        """register_builtin_models passes the Sonnet 5 Bedrock model string used in models.yml to LiteLLM."""
        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
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

    def test_no_env_var_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the env var is unset, ``register_model`` is not called."""
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            register_external_models()
            mock_register.assert_not_called()

    def test_empty_env_var_is_noop(self) -> None:
        """An empty string env var is treated as unset."""
        with patch.dict("os.environ", {ENV_VAR_NAME: ""}):
            with patch(
                "ai_gateway.models.v2.litellm_model_registry.register_model"
            ) as mock_register:
                register_external_models()
                mock_register.assert_not_called()

    def test_explicit_file_path_registers_models(self, tmp_path: Path) -> None:
        """An explicit ``file_path`` argument loads and registers models."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            register_external_models(file_path=str(file_path))

            mock_register.assert_called_once()
            registered = mock_register.call_args.args[0]
            assert len(registered) == 2
            assert "fireworks_ai/accounts/gitlab/deployments/test-model" in registered

    def test_env_var_file_path_registers_models(self, tmp_path: Path) -> None:
        """The env var is honored when no explicit path is given."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)

        with patch.dict("os.environ", {ENV_VAR_NAME: str(file_path)}):
            with patch(
                "ai_gateway.models.v2.litellm_model_registry.register_model"
            ) as mock_register:
                register_external_models()
                mock_register.assert_called_once()

    def test_missing_file_logs_warning_and_continues(self, tmp_path: Path) -> None:
        """Missing files do not raise; ``register_model`` is not called."""
        missing = tmp_path / "missing.json"

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            # Should not raise
            register_external_models(file_path=str(missing))
            mock_register.assert_not_called()

    def test_invalid_json_logs_warning_and_continues(self, tmp_path: Path) -> None:
        """Invalid JSON does not raise; ``register_model`` is not called."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text('{"models": {"foo": [unclosed')

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            register_external_models(file_path=str(file_path))
            mock_register.assert_not_called()

    def test_invalid_structure_logs_warning_and_continues(self, tmp_path: Path) -> None:
        """Structural validation errors do not raise; registration is skipped."""
        file_path = tmp_path / "bad_structure.json"
        file_path.write_text(json.dumps({"models": {"my-model": "not-a-mapping"}}))

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            register_external_models(file_path=str(file_path))
            mock_register.assert_not_called()

    def test_empty_models_section_does_not_call_register(self, tmp_path: Path) -> None:
        """An empty ``models`` section results in no registration call."""
        file_path = tmp_path / "empty.json"
        file_path.write_text(json.dumps({"models": {}}))

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model"
        ) as mock_register:
            register_external_models(file_path=str(file_path))
            mock_register.assert_not_called()

    def test_register_model_exception_does_not_propagate(self, tmp_path: Path) -> None:
        """If ``register_model`` raises, the application still starts."""
        file_path = tmp_path / "models.json"
        file_path.write_text(VALID_JSON)

        with patch(
            "ai_gateway.models.v2.litellm_model_registry.register_model",
            side_effect=RuntimeError("litellm internal error"),
        ):
            # Should not raise
            register_external_models(file_path=str(file_path))


class TestIntegrationWithLiteLLM:
    """Integration tests verifying real LiteLLM registration."""

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
