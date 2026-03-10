"""Tests for pricing multipliers verification script."""

# pylint: disable=import-outside-toplevel

from unittest.mock import MagicMock

import pytest

from ai_gateway.model_selection.models import ChatAnthropicParams, ChatLiteLLMParams
from scripts.verify_pricing_multipliers import (
    get_pricing_keys,
    get_proxy_models,
    get_selectable_models,
    normalize_model_id,
)


class TestNormalizeModelId:
    """Test model ID normalization for CustomersDot compatibility."""

    def test_converts_at_symbol_to_dash(self):
        """Vertex models use @ which CustomersDot converts to -."""
        assert (
            normalize_model_id("claude-sonnet-4-5@20250929")
            == "claude-sonnet-4-5-20250929"
        )

    def test_preserves_ids_without_at_symbol(self):
        """Non-Vertex models should pass through unchanged."""
        assert normalize_model_id("gpt-5-codex") == "gpt-5-codex"

    def test_preserves_bedrock_format(self):
        """Bedrock model IDs have special format that should be unchanged."""
        model_id = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
        assert normalize_model_id(model_id) == model_id


class TestGetPricingKeys:
    """Test extraction of pricing multiplier keys from config."""

    def test_extracts_keys_from_valid_config(self):
        config = {
            "default": {
                "agent_llm_request": {
                    "resource_multipliers": {
                        "claude-sonnet-4-5-20250929": [
                            {"end_date": None, "multiplier": 1.0}
                        ],
                        "gpt-5-codex": [{"end_date": None, "multiplier": 0.6}],
                    }
                }
            }
        }
        keys = get_pricing_keys(config)
        assert keys == {"claude-sonnet-4-5-20250929", "gpt-5-codex"}

    def test_returns_empty_set_when_no_multipliers(self):
        config = {"default": {"agent_llm_request": {"resource_multipliers": {}}}}
        assert get_pricing_keys(config) == set()

    def test_exits_on_invalid_structure(self):
        with pytest.raises(SystemExit):
            get_pricing_keys({"default": {}})


class TestGetSelectableModels:
    """Test extraction of selectable models from unit_primitives.yml."""

    def test_extracts_and_normalizes_models(self, monkeypatch):
        """Should extract params.model and normalize @ to -."""
        mock_llm_def = MagicMock()
        mock_llm_def.params = ChatLiteLLMParams(model="claude-sonnet-4-5@20250929")

        mock_up_config = MagicMock()
        mock_up_config.feature_setting = "duo_chat"
        mock_up_config.default_model = "claude_sonnet_vertex"
        mock_up_config.selectable_models = []
        mock_up_config.dev = None

        mock_config = MagicMock()
        mock_config.get_llm_definitions.return_value = {
            "claude_sonnet_vertex": mock_llm_def
        }
        mock_config.get_unit_primitive_config.return_value = [mock_up_config]

        from ai_gateway.model_selection import ModelSelectionConfig

        monkeypatch.setattr(ModelSelectionConfig, "instance", lambda: mock_config)

        result = get_selectable_models()

        assert "claude-sonnet-4-5-20250929" in result
        assert result["claude-sonnet-4-5-20250929"] == "claude_sonnet_vertex"

    def test_includes_dev_selectable_models(self, monkeypatch):
        """Dev models should also be checked for pricing."""
        mock_llm_def = MagicMock()
        mock_llm_def.params = ChatLiteLLMParams(model="experimental-model")

        mock_dev = MagicMock()
        mock_dev.selectable_models = ["experimental_id"]

        mock_up_config = MagicMock()
        mock_up_config.feature_setting = "duo_chat"
        mock_up_config.default_model = "experimental_id"
        mock_up_config.selectable_models = []
        mock_up_config.dev = mock_dev

        mock_config = MagicMock()
        mock_config.get_llm_definitions.return_value = {"experimental_id": mock_llm_def}
        mock_config.get_unit_primitive_config.return_value = [mock_up_config]

        from ai_gateway.model_selection import ModelSelectionConfig

        monkeypatch.setattr(ModelSelectionConfig, "instance", lambda: mock_config)

        result = get_selectable_models()

        assert "experimental-model" in result

    def test_excludes_code_completions_models(self, monkeypatch):
        """code_completions uses flat rate pricing, not model-based multipliers."""
        mock_codestral = MagicMock()
        mock_codestral.params = ChatLiteLLMParams(model="codestral-2508")

        mock_claude = MagicMock()
        mock_claude.params = ChatAnthropicParams(model="claude-sonnet-4")

        mock_code_completions_config = MagicMock()
        mock_code_completions_config.feature_setting = "code_completions"
        mock_code_completions_config.default_model = "codestral_id"
        mock_code_completions_config.selectable_models = []
        mock_code_completions_config.dev = None

        mock_duo_chat_config = MagicMock()
        mock_duo_chat_config.feature_setting = "duo_chat"
        mock_duo_chat_config.default_model = "claude_id"
        mock_duo_chat_config.selectable_models = []
        mock_duo_chat_config.dev = None

        mock_config = MagicMock()
        mock_config.get_llm_definitions.return_value = {
            "codestral_id": mock_codestral,
            "claude_id": mock_claude,
        }
        mock_config.get_unit_primitive_config.return_value = [
            mock_code_completions_config,
            mock_duo_chat_config,
        ]

        from ai_gateway.model_selection import ModelSelectionConfig

        monkeypatch.setattr(ModelSelectionConfig, "instance", lambda: mock_config)

        result = get_selectable_models()

        assert "codestral-2508" not in result
        assert "claude-sonnet-4" in result


class TestGetProxyModels:
    """Test extraction of proxy models from models.yml."""

    def test_extracts_models_with_proxy_provider(self, monkeypatch):
        """Should extract models that have proxy_provider set."""
        mock_anthropic_model = MagicMock()
        mock_anthropic_model.proxy_provider = "anthropic"
        mock_anthropic_model.params = MagicMock()
        mock_anthropic_model.params.model = "claude-sonnet-4-5-20250929"

        mock_openai_model = MagicMock()
        mock_openai_model.proxy_provider = "openai"
        mock_openai_model.params = MagicMock()
        mock_openai_model.params.model = "gpt-5-codex"

        mock_config = MagicMock()
        mock_config.get_llm_definitions.return_value = {
            "claude_sonnet_4_5_20250929": mock_anthropic_model,
            "gpt_5_codex": mock_openai_model,
        }

        from ai_gateway.model_selection import ModelSelectionConfig

        monkeypatch.setattr(ModelSelectionConfig, "instance", lambda: mock_config)

        result = get_proxy_models()

        assert "claude-sonnet-4-5-20250929" in result
        assert result["claude-sonnet-4-5-20250929"] == "claude_sonnet_4_5_20250929"
        assert "gpt-5-codex" in result
        assert result["gpt-5-codex"] == "gpt_5_codex"

    def test_excludes_models_without_proxy_provider(self, monkeypatch):
        """Should not include models without proxy_provider."""
        mock_proxy_model = MagicMock()
        mock_proxy_model.proxy_provider = "anthropic"
        mock_proxy_model.params = MagicMock()
        mock_proxy_model.params.model = "claude-proxy"

        mock_non_proxy_model = MagicMock()
        mock_non_proxy_model.proxy_provider = None
        mock_non_proxy_model.params = MagicMock()
        mock_non_proxy_model.params.model = "claude-vertex"

        mock_config = MagicMock()
        mock_config.get_llm_definitions.return_value = {
            "claude_proxy": mock_proxy_model,
            "claude_vertex": mock_non_proxy_model,
        }

        from ai_gateway.model_selection import ModelSelectionConfig

        monkeypatch.setattr(ModelSelectionConfig, "instance", lambda: mock_config)

        result = get_proxy_models()

        assert "claude-proxy" in result
        assert "claude-vertex" not in result

    def test_normalizes_vertex_style_model_ids(self, monkeypatch):
        """Should normalize @ to - in model IDs."""
        mock_model = MagicMock()
        mock_model.proxy_provider = "anthropic"
        mock_model.params = MagicMock()
        mock_model.params.model = "claude-sonnet-4-5@20250929"

        mock_config = MagicMock()
        mock_config.get_llm_definitions.return_value = {
            "claude_sonnet_vertex": mock_model,
        }

        from ai_gateway.model_selection import ModelSelectionConfig

        monkeypatch.setattr(ModelSelectionConfig, "instance", lambda: mock_config)

        result = get_proxy_models()

        assert "claude-sonnet-4-5-20250929" in result
        assert "claude-sonnet-4-5@20250929" not in result
