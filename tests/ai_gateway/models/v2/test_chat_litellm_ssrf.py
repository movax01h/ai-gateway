# pylint: disable=file-naming-for-tests
import pytest

from ai_gateway.models.v2.chat_litellm import ChatLiteLLM


class TestChatLiteLLMSSRFProtection:
    """Test SSRF protection for ChatLiteLLM model."""

    def test_bind_rejects_api_base_when_custom_models_disabled(self):
        """Test that .bind() rejects api_base when custom_models_enabled=False."""
        model = ChatLiteLLM(
            model="gpt-4",
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_base="http://internal-server.local/ssrf")

    def test_bind_rejects_api_key_when_custom_models_disabled(self):
        """Test that .bind() rejects api_key when custom_models_enabled=False."""
        model = ChatLiteLLM(
            model="gpt-4",
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_key="stolen-key")

    def test_bind_allows_api_base_when_custom_models_enabled(self):
        """Test that .bind() allows api_base when custom_models_enabled=True."""
        model = ChatLiteLLM(
            model="gpt-4",
            custom_models_enabled=True,
        )

        bound = model.bind(api_base="http://custom-endpoint.com")
        assert bound is not None

    def test_bind_allows_api_key_when_custom_models_enabled(self):
        """Test that .bind() allows api_key when custom_models_enabled=True."""
        model = ChatLiteLLM(
            model="gpt-4",
            custom_models_enabled=True,
        )

        bound = model.bind(api_key="custom-key")
        assert bound is not None

    def test_bind_allows_fireworks_provider_when_custom_models_disabled(self):
        """Trusted provider (fireworks_ai) bypasses the SSRF check even when custom models are disabled."""
        model = ChatLiteLLM(model="gpt-4", custom_models_enabled=False)

        bound = model.bind(
            custom_llm_provider="fireworks_ai",
            api_base="https://api.fireworks.ai/inference/v1",
        )
        assert bound is not None

    def test_bind_allows_mistral_provider_when_custom_models_disabled(self):
        """Trusted provider (mistral) bypasses the SSRF check even when custom models are disabled."""
        model = ChatLiteLLM(model="devstral", custom_models_enabled=False)

        bound = model.bind(custom_llm_provider="mistral", api_key="mistral-key")
        assert bound is not None
