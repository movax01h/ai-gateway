import pytest
from pydantic import ValidationError

from ai_gateway.model_selection.models import (
    MAX_MODEL_RETRIES,
    AnthropicWorkloadIdentityParams,
    BaseModelParams,
    ChatAmazonQParams,
    ChatAnthropicParams,
    ChatGoogleGenAIParams,
    ChatLiteLLMParams,
    ChatOpenAIParams,
    CompletionLiteLLMParams,
    CompletionType,
    EmbeddingLiteLLMParams,
    OpenAIProviderParams,
    OpenAIReasoningParams,
    validate_client_headers,
)


class TestValidateClientHeaders:
    """Fail-closed header allowlist for model header maps."""

    @pytest.mark.parametrize(
        "headers",
        [
            None,
            {},
            {"anthropic-beta": "prompt-caching-2024-07-31"},
            {"anthropic-version": "2023-06-01"},
            {"X-Trace-Id": "abc"},
            {"x-trace-id": "abc", "anthropic-beta": "x"},
        ],
    )
    def test_allows_allowlisted_headers(self, headers):
        assert validate_client_headers(headers) == headers

    @pytest.mark.parametrize(
        "forbidden",
        [
            "Host",
            "host",
            "HOST",
            "  Host  ",
            "X-Goog-User-Project",
            ":authority",
            "Authorization",
            "Proxy-Authorization",
            "Cookie",
            "X-Custom-Header",
            "User-Agent",
            "Content-Type",
        ],
    )
    def test_rejects_non_allowlisted_headers(self, forbidden):
        with pytest.raises(ValueError, match="is not permitted"):
            validate_client_headers({forbidden.strip(): "value"})

    def test_rejects_when_one_of_many_is_not_allowlisted(self):
        with pytest.raises(ValueError, match="Host"):
            validate_client_headers({"anthropic-beta": "ok", "Host": "example.test"})

    def test_additional_allowed_permits_extra_keys(self):
        headers = {"X-Api-Subscription": "uuid", "anthropic-beta": "x"}

        assert (
            validate_client_headers(headers, additional_allowed=["X-Api-Subscription"])
            == headers
        )

    def test_additional_allowed_does_not_permit_other_keys(self):
        with pytest.raises(ValueError, match="is not permitted"):
            validate_client_headers(
                {"X-Other": "v"}, additional_allowed=["X-Api-Subscription"]
            )


class TestOpenAIReasoningParams:
    def test_rejects_empty_block(self):
        with pytest.raises(ValidationError, match="at least one of effort/summary"):
            OpenAIReasoningParams()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"effort": "high"},
            {"effort": 8},
            {"summary": "auto"},
            {"effort": "low", "summary": "auto"},
        ],
    )
    def test_accepts_partial_or_full_block(self, kwargs):
        params = OpenAIReasoningParams(**kwargs)

        assert params.model_dump(exclude_none=True) == kwargs

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            OpenAIReasoningParams(effort="high", budget_tokens=1024)


class TestChatOpenAIParams:
    def test_inherits_mixin_fields_and_strict_config(self):
        assert issubclass(ChatOpenAIParams, OpenAIProviderParams)
        assert ChatOpenAIParams.model_config["extra"] == "forbid"
        assert ChatOpenAIParams.model_config["protected_namespaces"] == ()

        with pytest.raises(ValidationError):
            ChatOpenAIParams(bogus_field=1)

    def test_all_optional_fields_unset_is_valid(self):
        params = ChatOpenAIParams(max_tokens=100)

        assert params.verbosity is None
        assert params.reasoning is None

    def test_dump_merges_base_and_provider_fields(self):
        params = ChatOpenAIParams(
            max_tokens=100,
            verbosity="low",
            reasoning={"summary": "auto", "effort": 8},
        )

        assert params.model_dump(exclude_none=True) == {
            "max_tokens": 100,
            "max_retries": 1,
            "verbosity": "low",
            "reasoning": {"summary": "auto", "effort": 8},
        }


class TestChatAnthropicParams:
    WIF = {
        "federation_rule_id": "fdrl_test",
        "organization_id": "00000000-0000-0000-0000-000000000000",
        "service_account_id": "svac_test",
    }

    def test_workload_identity_is_optional_and_nested(self):
        params = ChatAnthropicParams(
            model="claude-fable-5-1", workload_identity=self.WIF
        )

        assert isinstance(params.workload_identity, AnthropicWorkloadIdentityParams)
        assert params.workload_identity.audience == "https://api.anthropic.com"
        assert params.workload_identity.workspace_id is None
        assert ChatAnthropicParams(model="claude-fable-5-1").workload_identity is None

    def test_workload_identity_is_hashable(self):
        a = AnthropicWorkloadIdentityParams(**self.WIF)
        b = AnthropicWorkloadIdentityParams(**self.WIF)

        assert hash(a) == hash(b)
        with pytest.raises(ValidationError):
            a.federation_rule_id = "fdrl_other"

    def test_workload_identity_rejects_unknown_fields(self):
        with pytest.raises(ValidationError, match="identity_token_file"):
            AnthropicWorkloadIdentityParams(**self.WIF, identity_token_file="/token")

    @pytest.mark.parametrize(
        "field", ["federation_rule_id", "organization_id", "service_account_id"]
    )
    def test_workload_identity_requires_token_exchange_fields(self, field):
        wif = {k: v for k, v in self.WIF.items() if k != field}

        with pytest.raises(ValidationError, match=field):
            AnthropicWorkloadIdentityParams(**wif)

    def test_api_key_and_workload_identity_are_mutually_exclusive(self):
        with pytest.raises(ValidationError, match="mutually exclusive"):
            ChatAnthropicParams(
                model="claude-fable-5-1", api_key="sk-ant-x", workload_identity=self.WIF
            )


# Each params class with the fields it requires besides max_retries.
_PARAMS_CLASSES = [
    (BaseModelParams, {}),
    (ChatAmazonQParams, {}),
    (ChatAnthropicParams, {}),
    (ChatGoogleGenAIParams, {}),
    (ChatLiteLLMParams, {}),
    (ChatOpenAIParams, {}),
    (CompletionLiteLLMParams, {"completion_type": CompletionType.TEXT}),
    (EmbeddingLiteLLMParams, {}),
]


class TestMaxRetriesCap:
    def test_covers_every_params_class(self):
        covered = {params_class for params_class, _ in _PARAMS_CLASSES}

        assert set(BaseModelParams.__subclasses__()) <= covered

    @pytest.mark.parametrize(("params_class", "required"), _PARAMS_CLASSES)
    @pytest.mark.parametrize("max_retries", [None, 0, 1, MAX_MODEL_RETRIES])
    def test_accepts_values_within_cap(self, params_class, required, max_retries):
        params = params_class(**required, max_retries=max_retries)

        assert params.max_retries == max_retries

    @pytest.mark.parametrize(("params_class", "required"), _PARAMS_CLASSES)
    @pytest.mark.parametrize("max_retries", [-1, MAX_MODEL_RETRIES + 1])
    def test_rejects_values_outside_cap(self, params_class, required, max_retries):
        with pytest.raises(ValidationError, match="max_retries"):
            params_class(**required, max_retries=max_retries)
