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

    def test_bind_allows_fireworks_provider_on_allowlisted_api_base(self):
        """Managed Fireworks: the operator-configured endpoint is allowlisted, so the server key passes."""
        model = ChatLiteLLM(
            model="gpt-4",
            custom_models_enabled=False,
            allowed_api_bases=frozenset(["https://api.fireworks.ai/inference/v1"]),
        )

        bound = model.bind(
            custom_llm_provider="fireworks_ai",
            api_base="https://api.fireworks.ai/inference/v1",
            api_key="fireworks-provider-key",
        )
        assert bound is not None

    def test_bind_allows_mistral_provider_key_without_api_base(self):
        """Managed Mistral: a trusted provider waives the api_key check, and sends no endpoint."""
        model = ChatLiteLLM(model="devstral", custom_models_enabled=False)

        bound = model.bind(custom_llm_provider="mistral", api_key="mistral-key")
        assert bound is not None

    @pytest.mark.parametrize("provider", ["fireworks_ai", "mistral"])
    @pytest.mark.parametrize("api_key", [None, "custom-key"])
    def test_bind_rejects_non_allowlisted_api_base_for_trusted_provider(
        self, provider, api_key
    ):
        """A trusted provider waives the api_key check only: api_base still has to be allowlisted."""
        model = ChatLiteLLM(
            model="gpt-4",
            custom_models_enabled=False,
            allowed_api_bases=frozenset(["https://api.fireworks.ai/inference/v1"]),
        )

        with pytest.raises(ValueError, match="api_base is not allowed"):
            model.bind(
                custom_llm_provider=provider,
                api_base="http://not-allowed.example.com",
                api_key=api_key,
            )

    def test_bind_tools_rejects_api_base_when_custom_models_disabled(self):
        """.bind_tools() rejects a client-supplied api_base when custom models are disabled."""
        model = ChatLiteLLM(model="gpt-4", custom_models_enabled=False)

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.bind_tools([], api_base="http://client-endpoint.example.com")

    def test_validate_endpoint_kwargs_rejects_api_base_when_custom_models_disabled(
        self,
    ):
        """The shared guard rejects api_base regardless of which bind path invokes it."""
        model = ChatLiteLLM(model="gpt-4", custom_models_enabled=False)

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.validate_endpoint_kwargs(
                {"api_base": "http://client-endpoint.example.com"}
            )

    def test_tool_bound_wrapper_bind_does_not_validate_so_base_guard_is_required(self):
        """A tool-bound RunnableBinding wrapper's .bind() does not run the guard, so callers must run
        validate_endpoint_kwargs on the base model before the final bind."""
        model = ChatLiteLLM(model="gpt-4", custom_models_enabled=False)

        wrapper = model.bind_tools([])
        client_kwargs = {"api_base": "http://client-endpoint.example.com"}

        wrapper.bind(**client_kwargs)

        with pytest.raises(
            ValueError, match="specifying custom models endpoint is disabled"
        ):
            model.validate_endpoint_kwargs(client_kwargs)
