# pylint: disable=file-naming-for-tests
import pytest
from google.genai import Client

from ai_gateway.models.v2.chat_google_genai import ChatGoogleGenerativeAI


class TestChatGoogleGenerativeAISSRFProtection:
    """Test SSRF protection for ChatGoogleGenerativeAI model."""

    @pytest.fixture(name="google_client")
    def google_client_fixture(self):
        return Client(api_key="test-key")

    def test_bind_rejects_api_base_when_custom_models_disabled(self, google_client):
        """Test that .bind() rejects api_base when custom_models_enabled=False."""
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="dummy",
            client=google_client,
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_base="http://internal-server.local/ssrf")

    def test_bind_rejects_api_key_when_custom_models_disabled(self, google_client):
        """Test that .bind() rejects api_key when custom_models_enabled=False."""
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="dummy",
            client=google_client,
            custom_models_enabled=False,
        )

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind(api_key="stolen-key")

    def test_bind_allows_api_base_when_custom_models_enabled(self, google_client):
        """Test that .bind() allows api_base when custom_models_enabled=True."""
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="dummy",
            client=google_client,
            custom_models_enabled=True,
        )

        bound = model.bind(api_base="http://custom-endpoint.com")
        assert bound is not None

    def test_bind_allows_api_key_when_custom_models_enabled(self, google_client):
        """Test that .bind() allows api_key when custom_models_enabled=True."""
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key="dummy",
            client=google_client,
            custom_models_enabled=True,
        )

        bound = model.bind(api_key="custom-key")
        assert bound is not None
