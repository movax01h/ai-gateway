# pylint: disable=file-naming-for-tests
import pytest

from ai_gateway.models.v2.openai import ChatOpenAI


class TestChatOpenAISSRFProtection:
    """Test SSRF protection for ChatOpenAI model."""

    def test_bind_rejects_api_base_when_custom_models_disabled(self):
        """Test that .bind() rejects api_base when custom_models_enabled=False."""
        model = ChatOpenAI(
            model="gpt-4",
            openai_api_key="dummy",
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_base="http://internal-server.local/ssrf")

    def test_bind_rejects_api_key_when_custom_models_disabled(self):
        """Test that .bind() rejects api_key when custom_models_enabled=False."""
        model = ChatOpenAI(
            model="gpt-4",
            openai_api_key="dummy",
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_key="stolen-key")

    def test_bind_allows_api_base_when_custom_models_enabled(self):
        """Test that .bind() allows api_base when custom_models_enabled=True."""
        model = ChatOpenAI(
            model="gpt-4",
            openai_api_key="dummy",
            custom_models_enabled=True,
        )

        bound = model.bind(api_base="http://custom-endpoint.com")
        assert bound is not None

    def test_bind_allows_api_key_when_custom_models_enabled(self):
        """Test that .bind() allows api_key when custom_models_enabled=True."""
        model = ChatOpenAI(
            model="gpt-4",
            openai_api_key="dummy",
            custom_models_enabled=True,
        )

        bound = model.bind(api_key="custom-key")
        assert bound is not None
