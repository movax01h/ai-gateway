from unittest import mock
from unittest.mock import MagicMock, patch

import httpx
import pytest
from anthropic import AsyncAnthropic
from httpx import AsyncClient, Limits
from structlog.testing import capture_logs

from ai_gateway.config import ConfigBedrockGuardrail
from ai_gateway.models import ModelMetadata
from ai_gateway.models.base import (
    init_anthropic_client,
    log_request,
    validate_custom_endpoint,
)
from ai_gateway.models.base_text import TextGenModelBase
from ai_gateway.models.guardrails import BEDROCK_GUARDRAIL_PROVIDERS


@pytest.mark.asyncio
async def test_init_anthropic_client():
    with patch("ai_gateway.models.base._DefaultAsyncHttpxClient") as mock_client:
        mock_http_client = MagicMock(spec=AsyncClient)
        mock_client.return_value = mock_http_client

        client = init_anthropic_client()

        assert isinstance(client, AsyncAnthropic)
        mock_client.assert_called_once()

        limits_arg = mock_client.call_args[1]["limits"]
        assert isinstance(limits_arg, Limits)
        assert limits_arg.max_connections == 1000
        assert limits_arg.max_keepalive_connections == 100
        assert limits_arg.keepalive_expiry == 30


# httpx attaches the effective per-request timeout as an extension; this is the
# ground truth the HTTP layer (and litellm's aiohttp transport, via the `read`
# component) enforces.
@pytest.mark.parametrize(
    ("extensions", "expected_timeout"),
    [
        (
            {
                "timeout": {
                    "connect": 5.0,
                    "read": 30.0,
                    "write": 30.0,
                    "pool": 30.0,
                }
            },
            {"connect": 5.0, "read": 30.0, "write": 30.0, "pool": 30.0},
        ),
        ({}, None),
    ],
)
class TestLogRequest:
    @pytest.mark.asyncio
    async def test_logs_request_timeout(
        self, extensions: dict, expected_timeout: dict | None
    ):
        request = httpx.Request(
            "POST",
            "http://example.com/v1/messages",
            content=b"{}",
            extensions=extensions,
        )

        with patch("ai_gateway.models.base.can_log_request_data", return_value=False):
            with capture_logs() as cap_logs:
                await log_request(request)

        entry = next(
            log_entry
            for log_entry in cap_logs
            if log_entry["event"] == "Request to LLM"
        )
        assert entry["request_timeout"] == expected_timeout

    @pytest.mark.asyncio
    async def test_logs_request_timeout_with_request_data_logging(
        self, extensions: dict, expected_timeout: dict | None
    ):
        """The request-data-enabled branch logs via `request_log`, whose processors drop events outside a request
        context, so assert on the logger call directly."""
        request = httpx.Request(
            "POST",
            "http://example.com/v1/messages",
            content=b"{}",
            extensions=extensions,
        )

        with patch("ai_gateway.models.base.can_log_request_data", return_value=True):
            with patch("ai_gateway.models.base.request_log") as mock_request_log:
                await log_request(request)

        mock_request_log.info.assert_called_once()
        assert (
            mock_request_log.info.call_args.kwargs["request_timeout"]
            == expected_timeout
        )


class TestTextGenBaseModel:
    class TestClass(TextGenModelBase):
        @property
        def metadata(self):
            return ModelMetadata(engine="vertex", name="codestral-2508")

        async def generate(self, **kwargs):
            pass

    @mock.patch("ai_gateway.models.base.config.model_engine_limits")
    def test_instrumentator(self, mock_config):
        mock_config.for_model.return_value = 7

        model = TestTextGenBaseModel.TestClass()
        instrumentator = model.instrumentator

        mock_config.for_model.assert_called_with(engine="vertex", name="codestral-2508")
        assert instrumentator.limits == 7


@pytest.mark.parametrize("provider", ["bedrock", "vertex_ai"])
def test_model_metadata_to_params_removes_api_base_for_specific_providers(provider):
    class TestClass(TextGenModelBase):
        def __init__(self, metadata):
            self._metadata = metadata

        @property
        def metadata(self):
            return self._metadata

        async def generate(self, **kwargs):
            pass

    model = TestClass(
        ModelMetadata(
            engine="litellm",
            name="model-a",
            endpoint="https://api.example.com",
            api_key="abcde",
            identifier=f"{provider}/model/identifier",
        )
    )

    assert model.model_metadata_to_params() == {
        "api_key": "abcde",
        "model": "model/identifier",
        "custom_llm_provider": provider,
    }


class TestValidateCustomEndpoint:
    def test_allows_when_custom_models_enabled(self):
        # Should not raise regardless of api_base/api_key
        validate_custom_endpoint(
            True,
            api_base="https://any.endpoint.example.com",
            api_key="sk-secret",
        )

    def test_allows_when_neither_api_base_nor_key(self):
        validate_custom_endpoint(False, api_base=None, api_key=None)

    def test_raises_for_api_base_not_in_allowed(self):
        with pytest.raises(ValueError, match="api_base is not allowed"):
            validate_custom_endpoint(
                False,
                api_base="https://custom.example.com",
                api_key=None,
            )

    def test_raises_for_api_key_without_api_base(self):
        with pytest.raises(ValueError, match="api_key is not allowed"):
            validate_custom_endpoint(False, api_base=None, api_key="sk-secret")

    def test_allows_api_base_in_allowed_api_bases(self):
        allowed = frozenset(["https://allowed.example.com"])
        validate_custom_endpoint(
            False,
            api_base="https://allowed.example.com",
            api_key=None,
            allowed_api_bases=allowed,
        )

    def test_allows_trailing_slash_normalized_match(self):
        allowed = frozenset(["https://allowed.example.com"])
        # caller passes with trailing slash — should still match
        validate_custom_endpoint(
            False,
            api_base="https://allowed.example.com/",
            api_key=None,
            allowed_api_bases=allowed,
        )

    def test_allows_trailing_slash_in_allowlist(self):
        allowed = frozenset(["https://allowed.example.com/"])
        # allowlist entry has trailing slash — caller without should still match
        validate_custom_endpoint(
            False,
            api_base="https://allowed.example.com",
            api_key=None,
            allowed_api_bases=allowed,
        )


class TestModelMetadataToParamsBedrockGuardrail:
    class _TestModel(TextGenModelBase):
        def __init__(self, metadata):
            self._metadata = metadata

        @property
        def metadata(self):
            return self._metadata

        async def generate(self, **kwargs):
            pass

    @pytest.fixture(name="guardrail_config")
    def guardrail_config_fixture(self):
        return ConfigBedrockGuardrail(
            guardrailIdentifier="abc123",
            guardrailVersion="1",
            trace="enabled",
        )

    @pytest.mark.parametrize("provider", sorted(BEDROCK_GUARDRAIL_PROVIDERS))
    def test_includes_guardrail_config_for_bedrock_providers(
        self, guardrail_config, provider
    ):
        model = self._TestModel(
            ModelMetadata(
                engine="litellm",
                name="model-a",
                api_key="abcde",
                identifier=f"{provider}/some-model",
            )
        )

        params = model.model_metadata_to_params(
            bedrock_guardrail_config=guardrail_config,
        )
        assert params["guardrailConfig"] == {
            "guardrailIdentifier": "abc123",
            "guardrailVersion": "1",
            "trace": "enabled",
        }

    def test_no_guardrail_config_for_non_bedrock_provider(self, guardrail_config):
        model = self._TestModel(
            ModelMetadata(
                engine="litellm",
                name="model-a",
                endpoint="https://api.example.com",
                api_key="abcde",
                identifier="anthropic/some-model",
            )
        )

        params = model.model_metadata_to_params(
            bedrock_guardrail_config=guardrail_config,
        )
        assert "guardrailConfig" not in params

    def test_no_guardrail_config_when_none(self):
        model = self._TestModel(
            ModelMetadata(
                engine="litellm",
                name="model-a",
                api_key="abcde",
                identifier="bedrock/some-model",
            )
        )

        params = model.model_metadata_to_params()
        assert "guardrailConfig" not in params

    def test_no_guardrail_config_without_identifier(self, guardrail_config):
        model = self._TestModel(
            ModelMetadata(
                engine="litellm",
                name="model-a",
                endpoint="https://api.example.com",
                api_key="abcde",
            )
        )

        params = model.model_metadata_to_params(
            bedrock_guardrail_config=guardrail_config,
        )
        assert "guardrailConfig" not in params

    def test_excludes_none_values_from_guardrail_config(self):
        config = ConfigBedrockGuardrail(guardrailIdentifier="abc123")
        model = self._TestModel(
            ModelMetadata(
                engine="litellm",
                name="model-a",
                api_key="abcde",
                identifier="bedrock/some-model",
            )
        )

        params = model.model_metadata_to_params(bedrock_guardrail_config=config)
        assert params["guardrailConfig"] == {
            "guardrailIdentifier": "abc123",
            "trace": "disabled",
        }
