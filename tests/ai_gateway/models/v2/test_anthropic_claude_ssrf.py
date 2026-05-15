# pylint: disable=file-naming-for-tests
import pytest
from anthropic import AsyncAnthropic

from ai_gateway.models.v2.anthropic_claude import ChatAnthropic


class TestChatAnthropicSSRFProtection:
    """Test SSRF protection for ChatAnthropic model."""

    @pytest.fixture(name="async_client")
    def async_client_fixture(self):
        return AsyncAnthropic(api_key="test-key")

    def test_bind_rejects_api_base_when_custom_models_disabled(self, async_client):
        """Test that .bind() rejects api_base when custom_models_enabled=False."""
        model = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key="dummy",
            async_client=async_client,
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_base="http://internal-server.local/ssrf")

    def test_bind_rejects_api_key_when_custom_models_disabled(self, async_client):
        """Test that .bind() rejects api_key when custom_models_enabled=False."""
        model = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key="dummy",
            async_client=async_client,
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_key="stolen-key")

    def test_bind_allows_api_base_when_custom_models_enabled(self, async_client):
        """Test that .bind() allows api_base when custom_models_enabled=True."""
        model = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key="dummy",
            async_client=async_client,
            custom_models_enabled=True,
        )

        bound = model.bind(api_base="http://custom-endpoint.com")
        assert bound is not None

    def test_bind_allows_api_key_when_custom_models_enabled(self, async_client):
        """Test that .bind() allows api_key when custom_models_enabled=True."""
        model = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key="dummy",
            async_client=async_client,
            custom_models_enabled=True,
        )

        bound = model.bind(api_key="custom-key")
        assert bound is not None
