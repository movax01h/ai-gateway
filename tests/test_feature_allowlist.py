import asyncio
import json
from datetime import date
from typing import Optional
from unittest.mock import Mock, patch

import pytest
import structlog.testing
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive

from ai_gateway.api.v1.embeddings.typing import EmbeddingsRequest
from ai_gateway.config import get_config
from ai_gateway.feature_allowlist import (
    ClientModelRequest,
    Surface,
    _AllowlistCheck,
    check_client_model_metadata,
    note_prompt_feature,
)
from ai_gateway.model_metadata import (
    FireworksModelMetadata,
    ModelMetadata,
    TypeModelMetadata,
    create_model_metadata,
)
from ai_gateway.model_selection import (
    FeatureSettingRequiredError,
    LLMDefinition,
    ModelNotAllowedForFeatureError,
    ModelSelectionConfig,
    ProviderNotServedError,
    UnitPrimitiveConfig,
)
from ai_gateway.model_selection.model_selection_config import (
    ChatAmazonQDefinition,
    ChatAmazonQParams,
    ChatLiteLLMDefinition,
    ChatLiteLLMParams,
    ModelTagEntry,
)
from ai_gateway.model_selection.types import (
    DefaultModelEntry,
    DevConfig,
    FeatureDeprecatedModel,
)
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from lib.context import StarletteUser, current_model_metadata_context

FEATURE = "test_feature"
OTHER_FEATURE = "other_feature"
FIREWORKS_FEATURE = "fireworks_feature"
MISMATCH = "Model allowlist mismatch"
NOTE = "Model allowlist note"
ALL_MODES = ["log", "fallback", "enforce"]
USER = StarletteUser(CloudConnectorUser(True, global_user_id="user-1"))


def _chat_model(identifier: str) -> ChatLiteLLMDefinition:
    return ChatLiteLLMDefinition(
        gitlab_identifier=identifier,
        name=identifier,
        max_context_tokens=200000,
        family=["mixtral"],
        params=ChatLiteLLMParams(model=f"vendor/{identifier}"),
    )


MODELS: dict[str, LLMDefinition] = {
    identifier: _chat_model(identifier)
    for identifier in [
        "allowed_model",
        "other_model",
        "dev_model",
        "deprecated_model",
        "tag_model",
        "custom_model",
    ]
}
MODELS["amazon_q"] = ChatAmazonQDefinition(
    gitlab_identifier="amazon_q",
    name="amazon_q",
    max_context_tokens=200000,
    family=["amazon_q"],
    params=ChatAmazonQParams(model="amazon_q"),
)
# GitLab-managed models whose builders set the identifier from the model definition.
MODELS["mistral_model"] = ChatLiteLLMDefinition(
    gitlab_identifier="mistral_model",
    name="mistral_model",
    max_context_tokens=200000,
    family=["mistral"],
    params=ChatLiteLLMParams(model="mistral-large", custom_llm_provider="mistral"),
)
# A model with distinctive limits, for the identifier-match copy in `_create_gitlab_metadata`.
MODELS["matched_model"] = ChatLiteLLMDefinition(
    gitlab_identifier="matched_model",
    name="matched_model",
    max_context_tokens=1_000_000,
    family=["claude"],
    params=ChatLiteLLMParams(model="claude-opus", max_tokens=64_000),
)
MODELS["fireworks_model"] = ChatLiteLLMDefinition(
    gitlab_identifier="fireworks_model",
    name="fireworks_model",
    max_context_tokens=200000,
    family=["kimi"],
    params=ChatLiteLLMParams(
        model="accounts/fireworks/models/kimi", custom_llm_provider="fireworks_ai"
    ),
)
# A Fireworks model without a model path, which only mock responses can serve.
MODELS["fireworks_mock_model"] = ChatLiteLLMDefinition(
    gitlab_identifier="fireworks_mock_model",
    name="fireworks_mock_model",
    max_context_tokens=200000,
    family=["kimi"],
    params=ChatLiteLLMParams(custom_llm_provider="fireworks_ai"),
)

FEATURE_CONFIG = UnitPrimitiveConfig(
    feature_setting=FEATURE,
    unit_primitives=[GitLabUnitPrimitive.DUO_CHAT],
    default_models=[DefaultModelEntry(identifier="allowed_model")],
    selectable_models=[
        "allowed_model",
        "deprecated_model",
        "mistral_model",
        "fireworks_model",
    ],
    models_for_tags={"small": ModelTagEntry(models=["tag_model"])},
    deprecated_models=[
        FeatureDeprecatedModel(
            identifier="deprecated_model",
            deprecation_date=date(2026, 1, 1),
            removal_version="19.9",
        )
    ],
    dev=DevConfig(selectable_models=["dev_model"]),
)
# A second feature whose default is a model the first feature does not allow.
OTHER_FEATURE_CONFIG = UnitPrimitiveConfig(
    feature_setting=OTHER_FEATURE,
    unit_primitives=[GitLabUnitPrimitive.DUO_CHAT],
    default_models=[DefaultModelEntry(identifier="other_model")],
    selectable_models=["other_model"],
)
# A feature whose default model is served by Fireworks.
FIREWORKS_FEATURE_CONFIG = UnitPrimitiveConfig(
    feature_setting=FIREWORKS_FEATURE,
    unit_primitives=[GitLabUnitPrimitive.COMPLETE_CODE],
    default_models=[DefaultModelEntry(identifier="fireworks_model")],
)

SELF_HOSTED_RAW = {
    "name": "custom_model",
    "endpoint": "https://self-hosted.internal/v1",
    "api_key": "admin-token",
    "identifier": "vendor-model",
}


@pytest.fixture(autouse=True)
def cloud_gateway(monkeypatch):
    """Run as GitLab's own gateway: a gateway that serves custom models skips the check."""
    monkeypatch.setattr(get_config().custom_models, "enabled", False)


@pytest.fixture(autouse=True)
def model_selection():
    def _get_model(name):
        if name in MODELS:
            return MODELS[name]
        raise ValueError(f"Invalid model identifier: {name}")

    with patch.multiple(
        ModelSelectionConfig,
        get_llm_definitions=Mock(return_value=MODELS),
        get_unit_primitive_config_map=Mock(
            return_value={
                FEATURE: FEATURE_CONFIG,
                OTHER_FEATURE: OTHER_FEATURE_CONFIG,
                FIREWORKS_FEATURE: FIREWORKS_FEATURE_CONFIG,
            }
        ),
        get_model=Mock(side_effect=_get_model),
    ):
        yield


def _dws_metadata(raw: dict) -> Optional[TypeModelMetadata]:
    """Resolve the gRPC header with the real Duo Workflow Service interceptor.

    The interceptor runs with the check off, so the test checks the metadata the interceptor resolved.
    """
    resolved = {}

    async def continuation(_):
        resolved["metadata"] = current_model_metadata_context.get()

    details = Mock(
        invocation_metadata=[
            (
                ModelMetadataInterceptor.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA,
                json.dumps(raw),
            )
        ]
    )
    token = current_user.set(USER.cloud_connector_user)
    try:
        with patch.object(
            get_config().model_selection, "feature_allowlist_mode_dws", "off"
        ):
            asyncio.run(
                ModelMetadataInterceptor().intercept_service(continuation, details)
            )
    finally:
        current_user.reset(token)
    return resolved["metadata"]


def _check(raw: dict, *, surface: Surface = "http", **kwargs):
    request = ClientModelRequest.from_raw(raw)
    if surface == "dws":
        metadata = _dws_metadata(raw)
        assert metadata is not None, (
            "the Duo Workflow Service does not resolve this shape"
        )
    else:
        metadata = create_model_metadata(dict(raw))
    with structlog.testing.capture_logs() as logs:
        result = check_client_model_metadata(
            request, metadata, surface=surface, **kwargs
        )
    return metadata, result, logs


def _reasons(logs, event=MISMATCH):
    return [entry["reason"] for entry in logs if entry["event"] == event]


class TestClientModelRequest:
    def test_reads_the_fields_a_client_may_send(self):
        request = ClientModelRequest.from_raw(
            {
                "provider": "gitlab",
                "identifier": "model_a",
                "name": "model_b",
                "feature_setting": "duo_chat",
                "endpoint": "https://example.com",
            }
        )

        assert request == ClientModelRequest(
            provider="gitlab",
            identifier="model_a",
            name="model_b",
            feature_setting="duo_chat",
        )

    def test_reads_a_pydantic_model(self):
        raw = EmbeddingsRequest.ModelMetadata(provider="gitlab", identifier="model_a")

        assert ClientModelRequest.from_raw(raw) == ClientModelRequest(
            provider="gitlab", identifier="model_a"
        )

    @pytest.mark.parametrize("raw", [None, "text", [], 42])
    def test_ignores_values_that_are_not_mappings(self, raw):
        assert ClientModelRequest.from_raw(raw) == ClientModelRequest()

    @pytest.mark.parametrize(
        ("raw", "pinned"),
        [
            ({"identifier": "model_a"}, True),
            ({"name": "model_a"}, True),
            ({"identifier": "", "feature_setting": "duo_chat"}, False),
            ({"identifier": None, "feature_setting": "duo_chat"}, False),
            ({"feature_setting": "duo_chat"}, False),
        ],
    )
    def test_pinned(self, raw, pinned):
        assert ClientModelRequest.from_raw(raw).pinned is pinned

    def test_truncates_long_values(self):
        request = ClientModelRequest.from_raw({"identifier": "x" * 300})

        assert request.identifier == "x" * 255


@pytest.mark.usefixtures("feature_allowlist_mode")
class TestCheckClientModelMetadata:
    @pytest.mark.parametrize("feature_allowlist_mode", ["off"], indirect=True)
    def test_off_mode_does_not_check(self):
        metadata, result, logs = _check(
            {
                "provider": "gitlab",
                "identifier": "other_model",
                "feature_setting": FEATURE,
            }
        )

        assert result is metadata
        assert logs == []

    @pytest.mark.parametrize(
        ("surface", "mode", "served"),
        [("http", "fallback", "allowed_model"), ("dws", "log", "other_model")],
    )
    def test_each_surface_uses_its_own_mode(self, monkeypatch, surface, mode, served):
        model_selection = get_config().model_selection
        monkeypatch.setattr(model_selection, "feature_allowlist_mode_http", "fallback")
        monkeypatch.setattr(model_selection, "feature_allowlist_mode_dws", "log")

        _, result, logs = _check(
            {
                "provider": "gitlab",
                "identifier": "other_model",
                "feature_setting": FEATURE,
            },
            surface=surface,
        )

        assert result.llm_definition.gitlab_identifier == served
        assert [e["mode"] for e in logs if e["event"] == MISMATCH] == [mode]

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    def test_a_gateway_that_serves_custom_models_does_not_check(self, monkeypatch):
        monkeypatch.setattr(get_config().custom_models, "enabled", True)

        metadata, result, logs = _check(
            {
                "provider": "gitlab",
                "identifier": "other_model",
                "feature_setting": FEATURE,
            }
        )

        assert result is metadata
        assert logs == []

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    def test_amazon_q_is_not_checked(self):
        metadata, result, logs = _check(
            {
                "provider": "amazon_q",
                "name": "amazon_q",
                "role_arn": "arn:aws:iam::123456789012:role/example-role",
            }
        )

        assert result is metadata
        assert logs == []

    class TestProviderNotServed:
        """A gateway without custom models serves a fixed set of providers per surface.

        Every other provider resolves a managed definition all the same. `fallback` and `enforce` refuse it;
        `log` records it and runs the remaining checks.
        """

        NOT_SERVED = {
            "http": [
                "openai",
                "litellm",
                "codestral",
                "custom_openai",
                "bedrock",
                "vertex_ai",
                "unknown_provider",
                "mistral",
            ],
            "dws": ["openai", "litellm", "anthropic", "unknown_provider"],
        }

        @staticmethod
        def _raw(provider):
            if provider == "mistral":
                return {"provider": provider, "name": "mistral_model"}
            return {"provider": provider, **SELF_HOSTED_RAW}

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        @pytest.mark.parametrize(
            ("surface", "provider"),
            [(s, p) for s, providers in NOT_SERVED.items() for p in providers],
        )
        def test_log_mode_keeps_the_request_and_warns(self, surface, provider):
            metadata, result, logs = _check(self._raw(provider), surface=surface)

            assert result is metadata
            assert _reasons(logs)[0] == "provider_not_served"

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        @pytest.mark.parametrize("surface", ["http", "dws"])
        def test_log_mode_runs_the_remaining_checks(self, surface):
            """The log shows what the request would meet once the provider is fixed."""
            raw = {
                "provider": "openai",
                "name": "other_model",
                "feature_setting": FEATURE,
            }

            metadata, result, logs = _check(raw, surface=surface)

            assert result is metadata
            assert _reasons(logs) == ["provider_not_served", "model_not_allowed"]

        @pytest.mark.parametrize(
            "feature_allowlist_mode", ["fallback", "enforce"], indirect=True
        )
        @pytest.mark.parametrize(
            ("surface", "provider"),
            [(s, p) for s, providers in NOT_SERVED.items() for p in providers],
        )
        def test_stricter_modes_reject(self, surface, provider):
            with pytest.raises(
                ProviderNotServedError,
                match=f"This gateway does not serve models from provider '{provider}'",
            ):
                _check(self._raw(provider), surface=surface)

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        def test_a_managed_model_behind_a_custom_provider_is_refused(self):
            """`{"provider": "openai", "name": <managed model>}` has no endpoint to validate and would otherwise run the
            managed model on GitLab's keys."""
            raw = {
                "provider": "openai",
                "name": "other_model",
                "feature_setting": FEATURE,
            }

            with pytest.raises(ProviderNotServedError):
                _check(raw)

    class TestLegacyCompletionsProviders:
        """`/v2/code/completions` builds metadata with the provider of its legacy `model_provider` field."""

        PROVIDERS = ["anthropic", "vertex-ai", "fireworks_ai"]

        @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
        @pytest.mark.parametrize("provider", PROVIDERS)
        def test_an_allowed_model_passes(self, provider):
            metadata, result, logs = _check(
                {"provider": provider, "name": "allowed_model"},
                feature_setting=FEATURE,
            )

            assert result is metadata
            assert _reasons(logs) == []

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        @pytest.mark.parametrize("provider", PROVIDERS)
        def test_a_model_that_is_not_allowed_is_rejected(self, provider):
            with pytest.raises(
                ModelNotAllowedForFeatureError,
                match="Model 'other_model' is not available for 'test_feature'",
            ):
                _check(
                    {"provider": provider, "name": "other_model"},
                    feature_setting=FEATURE,
                )

    @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
    @pytest.mark.parametrize("surface", ["http", "dws"])
    def test_allowed_pinned_model_passes(self, surface):
        metadata, result, logs = _check(
            {
                "provider": "gitlab",
                "identifier": "allowed_model",
                "feature_setting": FEATURE,
            },
            surface=surface,
        )

        assert result is metadata
        assert _reasons(logs) == []

    @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
    @pytest.mark.parametrize("call_site_feature", [None, FEATURE])
    def test_feature_default_passes(self, call_site_feature):
        metadata, result, logs = _check(
            {"provider": "gitlab", "feature_setting": FEATURE},
            feature_setting=call_site_feature,
        )

        assert result is metadata
        assert _reasons(logs) == []

    class TestFeatureDefaultOfAnotherFeature:
        """The client asks for another feature's default on a call site that knows its own feature."""

        RAW = {"provider": "gitlab", "feature_setting": OTHER_FEATURE}

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        def test_log_mode_keeps_the_client_default_and_warns(self):
            metadata, result, logs = _check(self.RAW, feature_setting=FEATURE)

            assert result is metadata
            assert result.llm_definition.gitlab_identifier == "other_model"
            assert _reasons(logs) == ["feature_mismatch"]
            entry = next(e for e in logs if e["event"] == MISMATCH)
            assert entry["feature_setting"] == FEATURE
            assert entry["client_feature_setting"] == OTHER_FEATURE
            assert entry["pinned"] is False

        @pytest.mark.parametrize(
            "feature_allowlist_mode", ["fallback", "enforce"], indirect=True
        )
        def test_stricter_modes_serve_the_call_site_default(self):
            metadata, result, logs = _check(self.RAW, feature_setting=FEATURE)

            assert result is not metadata
            assert result.llm_definition.gitlab_identifier == "allowed_model"
            assert _reasons(logs) == ["feature_mismatch"]

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        def test_a_pin_from_another_feature_is_checked_against_the_call_site(self):
            raw = {**self.RAW, "identifier": "other_model"}

            with pytest.raises(
                ModelNotAllowedForFeatureError,
                match="Model 'other_model' is not available for 'test_feature'",
            ):
                _check(raw, feature_setting=FEATURE)

    @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
    def test_tag_model_is_allowed(self):
        metadata, result, _ = _check(
            {
                "provider": "gitlab",
                "identifier": "tag_model",
                "feature_setting": FEATURE,
            }
        )

        assert result is metadata

    class TestModelNotAllowed:
        RAW = {
            "provider": "gitlab",
            "identifier": "other_model",
            "feature_setting": FEATURE,
        }

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        def test_log_mode_keeps_the_model_and_warns(self):
            metadata, result, logs = _check(
                self.RAW, log_context={"path": "/v1/prompts/chat/react"}
            )

            assert result is metadata
            assert _reasons(logs) == ["model_not_allowed"]
            entry = next(e for e in logs if e["event"] == MISMATCH)
            assert entry["log_level"] == "warning"
            assert entry["mode"] == "log"
            assert entry["surface"] == "http"
            assert entry["feature_setting"] == FEATURE
            assert entry["requested_name"] is None
            assert entry["requested_identifier"] == "other_model"
            assert entry["served_identifier"] == "other_model"
            assert entry["pinned"] is True
            assert entry["path"] == "/v1/prompts/chat/react"

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        def test_the_check_fields_win_over_the_caller_context(self):
            _, _, logs = _check(
                self.RAW,
                log_context={
                    "event": "caller event",
                    "reason": "caller reason",
                    "mode": "caller mode",
                    "path": "/v1/prompts/chat/react",
                },
            )

            assert logs == [
                {
                    "event": MISMATCH,
                    "log_level": "warning",
                    "reason": "model_not_allowed",
                    "mode": "log",
                    "surface": "http",
                    "feature_setting": FEATURE,
                    "client_feature_setting": FEATURE,
                    "provider": "gitlab",
                    "requested_name": None,
                    "requested_identifier": "other_model",
                    "pinned": True,
                    "served_identifier": "other_model",
                    "path": "/v1/prompts/chat/react",
                }
            ]

        @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
        def test_fallback_mode_serves_the_feature_default(self):
            _, result, logs = _check(self.RAW)

            assert isinstance(result, ModelMetadata)
            assert result.llm_definition.gitlab_identifier == "allowed_model"
            assert result.provider == "gitlab"
            assert _reasons(logs) == ["model_not_allowed"]

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        @pytest.mark.parametrize("surface", ["http", "dws"])
        def test_enforce_mode_rejects(self, surface):
            with pytest.raises(
                ModelNotAllowedForFeatureError,
                match="Model 'other_model' is not available for 'test_feature'",
            ):
                _check(self.RAW, surface=surface)

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        def test_the_call_site_feature_wins_over_the_client_value(self):
            raw = {"provider": "gitlab", "identifier": "other_model"}

            with pytest.raises(ModelNotAllowedForFeatureError):
                _check(raw, feature_setting=FEATURE)

    class TestFeatureNotUsable:
        """A named model whose feature is missing or unknown, so the allowlist cannot be applied."""

        MISSING_RAW = {"provider": "gitlab", "name": "other_model"}
        UNKNOWN_RAW = {
            "provider": "gitlab",
            "identifier": "other_model",
            "feature_setting": "nope",
        }
        CASES = [
            pytest.param(
                "http", MISSING_RAW, "missing_feature_setting", id="http-missing"
            ),
            pytest.param("http", UNKNOWN_RAW, "unknown_feature", id="http-unknown"),
            pytest.param("dws", UNKNOWN_RAW, "unknown_feature", id="dws-unknown"),
        ]
        ERRORS = [
            pytest.param(
                "http",
                MISSING_RAW,
                FeatureSettingRequiredError,
                "feature_setting is required to request model 'other_model'",
                id="http-missing",
            ),
            pytest.param(
                "http",
                UNKNOWN_RAW,
                ModelNotAllowedForFeatureError,
                "Model 'other_model' is not available for 'nope'",
                id="http-unknown",
            ),
            pytest.param(
                "dws",
                UNKNOWN_RAW,
                ModelNotAllowedForFeatureError,
                "Model 'other_model' is not available for 'nope'",
                id="dws-unknown",
            ),
        ]

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        @pytest.mark.parametrize(("surface", "raw", "reason"), CASES)
        def test_log_mode_keeps_the_model_and_warns(self, surface, raw, reason):
            metadata, result, logs = _check(raw, surface=surface)

            assert result is metadata
            assert _reasons(logs) == [reason]

        @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
        @pytest.mark.parametrize(("surface", "raw", "reason"), CASES)
        def test_fallback_mode_uses_the_prompt_default(self, surface, raw, reason):
            _, result, logs = _check(raw, surface=surface)

            assert result is None
            assert _reasons(logs) == [reason]

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        @pytest.mark.parametrize(("surface", "raw", "error", "message"), ERRORS)
        def test_enforce_mode_rejects(self, surface, raw, error, message):
            with pytest.raises(error, match=message):
                _check(raw, surface=surface)

        @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
        def test_dws_keeps_the_resume_replay_with_a_note(self):
            metadata, result, logs = _check(self.MISSING_RAW, surface="dws")

            assert result is metadata
            assert _reasons(logs) == []
            assert _reasons(logs, NOTE) == ["missing_feature_setting"]

    class TestIdentifierOverrideOnManagedModel:
        """`name` resolves an allowed model while the raw `identifier` rewrites the wire model."""

        RAW = {
            "provider": "gitlab",
            "name": "allowed_model",
            "identifier": "vertex_ai/other-model",
            "feature_setting": FEATURE,
        }

        @pytest.mark.parametrize("feature_allowlist_mode", ["log"], indirect=True)
        def test_log_mode_keeps_the_override_and_warns(self):
            metadata, result, logs = _check(self.RAW)

            assert result is metadata
            assert result.identifier == "vertex_ai/other-model"
            assert result.to_params()["model"] == "other-model"
            assert _reasons(logs) == ["managed_identifier_override"]

        @pytest.mark.parametrize(
            "feature_allowlist_mode", ["fallback", "enforce"], indirect=True
        )
        def test_stricter_modes_rebuild_the_model_from_its_definition(self):
            metadata, result, _ = _check(self.RAW)

            assert result is not metadata
            assert isinstance(result, ModelMetadata)
            assert result.llm_definition.gitlab_identifier == "allowed_model"
            assert result.identifier is None
            # No wire model override is left: the model comes from the definition.
            assert "model" not in result.to_params()

        @pytest.mark.parametrize(
            "feature_allowlist_mode", ["fallback", "enforce"], indirect=True
        )
        def test_stricter_modes_discard_limits_copied_from_the_identifier(self):
            """`_create_gitlab_metadata` copies limits and tuning from the model the identifier matched."""
            with patch.object(
                ModelSelectionConfig,
                "find_by_identifier",
                return_value=MODELS["matched_model"],
            ):
                metadata, result, _ = _check(self.RAW)

            assert metadata.llm_definition.max_context_tokens == 1_000_000
            assert result.llm_definition.gitlab_identifier == "allowed_model"
            assert result.llm_definition.max_context_tokens == 200000
            assert result.llm_definition.params.max_tokens is None

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        def test_a_client_endpoint_or_key_is_left_to_the_endpoint_check(self):
            """`validate_custom_endpoint` rejects both when the model is built."""
            metadata, result, logs = _check(
                {
                    "provider": "gitlab",
                    "identifier": "allowed_model",
                    "feature_setting": FEATURE,
                    "endpoint": "https://attacker.example.com/",
                    "api_key": "secret-key",
                }
            )

            assert result is metadata
            assert _reasons(logs) == []
            assert "secret-key" not in str(logs)
            assert "attacker.example.com" not in str(logs)

        @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
        def test_dws_checks_the_identifier_it_resolves(self):
            """The Duo Workflow Service resolves a `gitlab` request by `identifier` and ignores `name`."""
            metadata, result, logs = _check(
                {
                    "provider": "gitlab",
                    "name": "other_model",
                    "identifier": "allowed_model",
                    "feature_setting": FEATURE,
                },
                surface="dws",
            )

            assert result is metadata
            assert result.llm_definition.gitlab_identifier == "allowed_model"
            assert _reasons(logs) == []

        @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
        @pytest.mark.parametrize(
            ("raw", "carries_identifier"),
            [
                # The replay of a Mistral model: the Mistral builder sets `identifier`.
                (
                    {
                        "provider": "mistral",
                        "name": "mistral_model",
                        "identifier": "mistral/mistral-large",
                        "provider_keys": {"mistral_api_key": "server-key"},
                    },
                    True,
                ),
                # The replay of a Fireworks model carries `model_identifier`, not `identifier`.
                (
                    {
                        "provider": "fireworks_ai",
                        "name": "fireworks_model",
                        "model_identifier": "accounts/fireworks/models/kimi",
                        "fireworks_api_base_url": "https://fireworks.example.com/",
                        "provider_keys": {"fireworks_provider_api_key": "server-key"},
                    },
                    False,
                ),
            ],
        )
        def test_server_keyed_providers_are_not_flagged(self, raw, carries_identifier):
            """An `identifier` that the server builder sets is not a client override."""
            metadata, result, logs = _check(
                {**raw, "feature_setting": FEATURE}, surface="dws"
            )

            assert bool(getattr(metadata, "identifier", None)) is carries_identifier
            assert result is metadata
            assert _reasons(logs) == []

    class TestRebuiltMetadata:
        """Metadata that the check builds keeps the request settings that the server builders set."""

        RAW = {
            "provider": "fireworks_ai",
            "name": "other_model",
            "feature_setting": FIREWORKS_FEATURE,
        }

        @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
        def test_a_fireworks_default_keeps_the_session_and_cache(self):
            _, result, _ = _check({**self.RAW, "using_cache": False}, user=USER)

            assert isinstance(result, FireworksModelMetadata)
            assert result.llm_definition.gitlab_identifier == "fireworks_model"
            assert result.session_id == "user-1"
            assert result.using_cache is False

        @pytest.mark.parametrize("feature_allowlist_mode", ["fallback"], indirect=True)
        def test_a_fireworks_default_without_a_model_path_is_served_in_mock_mode(
            self, monkeypatch
        ):
            monkeypatch.setattr(get_config(), "mock_model_responses", True)

            with patch.object(
                ModelSelectionConfig,
                "get_model_for_feature",
                return_value=MODELS["fireworks_mock_model"],
            ):
                _, result, _ = _check(self.RAW)

            assert isinstance(result, FireworksModelMetadata)
            assert result.llm_definition.gitlab_identifier == "fireworks_mock_model"
            assert result.model_identifier == ""

    @pytest.mark.parametrize("feature_allowlist_mode", ["enforce"], indirect=True)
    @pytest.mark.parametrize(
        ("identifier", "reason"),
        [("dev_model", "dev_model"), ("deprecated_model", "deprecated_model")],
    )
    def test_dev_and_deprecated_models_are_allowed_with_a_note(
        self, identifier, reason
    ):
        metadata, result, logs = _check(
            {"provider": "gitlab", "identifier": identifier, "feature_setting": FEATURE}
        )

        assert result is metadata
        assert _reasons(logs) == []
        assert _reasons(logs, NOTE) == [reason]


def _note_prompt_feature(raw: dict, prompt_feature_setting: str):
    request = ClientModelRequest.from_raw(raw)
    metadata = create_model_metadata(dict(raw))
    with structlog.testing.capture_logs() as logs:
        note_prompt_feature(
            request,
            metadata,
            prompt_feature_setting=prompt_feature_setting,
            log_context={"prompt_id": "test/prompt"},
        )
    return logs


@pytest.mark.usefixtures("feature_allowlist_mode")
class TestNotePromptFeature:
    @pytest.mark.parametrize("feature_allowlist_mode", ALL_MODES, indirect=True)
    def test_notes_a_model_that_the_prompt_feature_does_not_allow(self):
        logs = _note_prompt_feature(
            {
                "provider": "gitlab",
                "identifier": "other_model",
                "feature_setting": OTHER_FEATURE,
            },
            FEATURE,
        )

        assert [
            (
                entry["reason"],
                entry["feature_setting"],
                entry["client_feature_setting"],
                entry["prompt_id"],
            )
            for entry in logs
        ] == [
            (
                "prompt_feature_model_not_allowed",
                FEATURE,
                OTHER_FEATURE,
                "test/prompt",
            )
        ]
        assert _reasons(logs) == []

    def test_notes_a_prompt_feature_that_does_not_exist(self):
        logs = _note_prompt_feature(
            {"provider": "gitlab", "identifier": "allowed_model"}, "no_such_feature"
        )

        assert _reasons(logs, NOTE) == ["prompt_feature_unknown"]

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param(
                {
                    "provider": "gitlab",
                    "identifier": "allowed_model",
                    "feature_setting": OTHER_FEATURE,
                },
                id="allowed-by-the-prompt-feature",
            ),
            pytest.param(
                {
                    "provider": "gitlab",
                    "identifier": "other_model",
                    "feature_setting": FEATURE,
                },
                id="client-sends-the-prompt-feature",
            ),
            pytest.param(
                {"provider": "gitlab", "feature_setting": OTHER_FEATURE},
                id="feature-default",
            ),
        ],
    )
    def test_notes_nothing(self, raw):
        assert _note_prompt_feature(raw, FEATURE) == []

    @pytest.mark.parametrize("feature_allowlist_mode", ["off"], indirect=True)
    def test_off_mode_notes_nothing(self):
        logs = _note_prompt_feature(
            {
                "provider": "gitlab",
                "identifier": "other_model",
                "feature_setting": OTHER_FEATURE,
            },
            FEATURE,
        )

        assert logs == []


def test_the_check_keeps_provider_keys_out_of_its_repr():
    check = _AllowlistCheck(
        mode="log",
        surface="http",
        request=ClientModelRequest(),
        feature=None,
        user=None,
        using_cache=None,
        log_context={},
        fireworks_api_base_url="",
        mock_model_responses=False,
        provider_keys={"mistral_api_key": "server-key"},
    )

    assert "server-key" not in repr(check)
