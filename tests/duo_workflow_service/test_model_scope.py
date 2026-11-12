import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from structlog.testing import capture_logs

from ai_gateway.config import get_config
from ai_gateway.model_metadata import FireworksModelMetadata, ModelMetadata
from ai_gateway.model_selection import ModelSelectionConfig
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)
from duo_workflow_service.model_scope import (
    FLAT_RATE_FLOW_FEATURE_SETTINGS,
    log_flow_model_feature,
    scope_flat_rate_flow_models,
)
from lib.billing_events import BILL_ONCE_PER_WORKFLOW_FEATURES
from lib.context.model import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)

FABLE = "claude_fable_5"
CODE_REVIEW = "code_review/v1"
REVIEW_FEATURE = "review_merge_request_dap"
NOT_ALLOWED = "Flat-rate flow model not allowed, switched to the feature default"
USER = CloudConnectorUser(
    authenticated=True,
    global_user_id="user-1",
    claims=UserClaims(scopes=["duo_agent_platform"], gitlab_realm="saas"),
)


@pytest.fixture(autouse=True)
def cloud_gateway(monkeypatch):
    """Run as GitLab's own gateway: a gateway that serves custom models keeps the client's models.

    The allowlist runs in `log` mode, so the header reaches the flat-rate check unchanged. The flat-rate check must
    hold on its own, also when the allowlist is rolled back to `log`.
    """
    monkeypatch.setattr(get_config().custom_models, "enabled", False)
    monkeypatch.setattr(
        get_config().model_selection, "feature_allowlist_mode_dws", "log"
    )


async def _resolve_header(raw: object) -> None:
    """Set the request's model metadata with the real Duo Workflow Service interceptor."""
    current_user.set(USER)
    details = MagicMock()
    details.invocation_metadata = [
        (
            ModelMetadataInterceptor.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA,
            json.dumps(raw),
        )
    ]
    await ModelMetadataInterceptor().intercept_service(AsyncMock(), details)


def _allowed(feature_setting: str) -> frozenset[str]:
    return ModelSelectionConfig.instance().allowed_models_for_feature(feature_setting)


def _default_of(feature_setting: str) -> list[str]:
    config = ModelSelectionConfig.instance().get_unit_primitive_config_map()
    return config[feature_setting].default_model_identifiers


def _served() -> tuple[str, list[str]]:
    default = current_model_metadata_context.get()
    by_tag = current_model_metadata_with_size_context.get()
    assert default is not None and by_tag is not None
    return (
        default.llm_definition.gitlab_identifier,
        [m.llm_definition.gitlab_identifier for m in by_tag.by_tag.values()],
    )


def test_every_flat_rate_flow_has_a_feature_setting():
    assert FLAT_RATE_FLOW_FEATURE_SETTINGS.keys() == BILL_ONCE_PER_WORKFLOW_FEATURES


@pytest.mark.parametrize("feature_setting", FLAT_RATE_FLOW_FEATURE_SETTINGS.values())
def test_every_feature_setting_exists(feature_setting):
    assert (
        feature_setting
        in ModelSelectionConfig.instance().get_unit_primitive_config_map()
    )


class TestModelNotAllowed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("flow", "feature_setting"),
        [
            (f"{flow}/v1", feature)
            for flow, feature in FLAT_RATE_FLOW_FEATURE_SETTINGS.items()
        ],
    )
    async def test_serves_the_default_of_the_flow_feature(self, flow, feature_setting):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})

        scope_flat_rate_flow_models(flow, USER)

        default, tags = _served()
        assert default in _default_of(feature_setting)
        assert set(tags) <= _allowed(feature_setting)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param({"provider": "gitlab", "identifier": FABLE}, id="identifier"),
            pytest.param(
                {
                    "provider": "gitlab",
                    "identifier": FABLE,
                    "feature_setting": "duo_agent_platform",
                },
                id="feature-that-allows-the-model",
            ),
            pytest.param(
                {"provider": "gitlab", "name": FABLE}, id="resume-replay-shape"
            ),
            pytest.param({"provider": "anthropic", "name": FABLE}, id="other-provider"),
        ],
    )
    async def test_the_client_cannot_choose_the_model(self, raw):
        assert FABLE in _allowed("duo_agent_platform")
        assert FABLE not in _allowed(REVIEW_FEATURE)
        await _resolve_header(raw)

        with capture_logs() as logs:
            scope_flat_rate_flow_models(CODE_REVIEW, USER)

        default, _ = _served()
        assert default in _default_of(REVIEW_FEATURE)
        assert [entry for entry in logs if entry["event"] == NOT_ALLOWED] == [
            {
                "event": NOT_ALLOWED,
                "log_level": "warning",
                "flow": CODE_REVIEW,
                "feature_setting": REVIEW_FEATURE,
                "requested_model": FABLE,
                "served_model": default,
            }
        ]

    @pytest.mark.asyncio
    async def test_a_feature_default_may_be_routed_within_the_flow_feature(self):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})

        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        assert (
            current_model_metadata_with_size_context.get().feature_setting
            == REVIEW_FEATURE
        )

    @pytest.mark.asyncio
    async def test_a_fireworks_default_gets_the_user_session(self):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})
        fireworks = ModelSelectionConfig.instance().get_model(
            "codestral_2508_fireworks"
        )

        with patch.object(
            ModelSelectionConfig, "get_model_for_feature", return_value=fireworks
        ):
            scope_flat_rate_flow_models(CODE_REVIEW, USER)

        default = current_model_metadata_context.get()
        assert isinstance(default, FireworksModelMetadata)
        assert default.session_id == "user-1"


class TestModelAllowed:
    @staticmethod
    def _allowed_model() -> str:
        """An allowed model of the review feature that is not its default."""
        return sorted(_allowed(REVIEW_FEATURE) - set(_default_of(REVIEW_FEATURE)))[0]

    @pytest.mark.asyncio
    async def test_keeps_the_client_model(self):
        model = self._allowed_model()
        await _resolve_header(
            {
                "provider": "gitlab",
                "identifier": model,
                "feature_setting": REVIEW_FEATURE,
            }
        )

        with capture_logs() as logs:
            scope_flat_rate_flow_models(CODE_REVIEW, USER)

        default, tags = _served()
        assert default == model
        assert set(tags) <= _allowed(REVIEW_FEATURE)
        assert [entry for entry in logs if entry["event"] == NOT_ALLOWED] == []

    @pytest.mark.asyncio
    async def test_a_pinned_model_is_not_routed(self):
        await _resolve_header(
            {
                "provider": "gitlab",
                "identifier": self._allowed_model(),
                "feature_setting": REVIEW_FEATURE,
            }
        )

        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        assert current_model_metadata_with_size_context.get().feature_setting is None

    @pytest.mark.asyncio
    async def test_a_raw_identifier_never_becomes_the_wire_model(self):
        """`name` resolves an allowed model, and the raw `identifier` next to it is ignored."""
        await _resolve_header(
            {
                "provider": "anthropic",
                "name": self._allowed_model(),
                "identifier": "vertex_ai/other-model",
            }
        )
        assert current_model_metadata_context.get().identifier is None

        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        default = current_model_metadata_context.get()
        assert isinstance(default, ModelMetadata)
        assert default.llm_definition.gitlab_identifier == self._allowed_model()
        assert default.identifier is None
        assert default.to_params().get("model") != "other-model"

    @pytest.mark.asyncio
    async def test_tags_come_from_the_flow_feature(self):
        await _resolve_header(
            {
                "provider": "gitlab",
                "identifier": self._allowed_model(),
                "feature_setting": "duo_agent_platform",
            }
        )
        features = ModelSelectionConfig.instance().get_unit_primitive_config_map()
        assert (
            current_model_metadata_with_size_context.get().by_tag.keys()
            == features["duo_agent_platform"].models_for_tags.keys()
            != features[REVIEW_FEATURE].models_for_tags.keys()
        )

        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        assert (
            current_model_metadata_with_size_context.get().by_tag.keys()
            == features[REVIEW_FEATURE].models_for_tags.keys()
        )


class TestNothingChanges:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flow", ["developer/v1", "chat", "software_development", "", None]
    )
    async def test_for_a_flow_that_is_not_billed_at_a_flat_rate(self, flow):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})
        before = current_model_metadata_with_size_context.get()

        scope_flat_rate_flow_models(flow, USER)

        assert current_model_metadata_with_size_context.get() is before
        assert (
            current_model_metadata_context.get().llm_definition.gitlab_identifier
            == FABLE
        )

    @pytest.mark.asyncio
    async def test_without_model_metadata(self):
        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        assert current_model_metadata_context.get() is None
        assert current_model_metadata_with_size_context.get() is None

    @pytest.mark.asyncio
    async def test_on_a_gateway_that_serves_custom_models(self, monkeypatch):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})
        monkeypatch.setattr(get_config().custom_models, "enabled", True)
        before = current_model_metadata_with_size_context.get()

        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        assert current_model_metadata_with_size_context.get() is before

    @pytest.mark.asyncio
    async def test_for_amazon_q(self):
        await _resolve_header(
            {
                "provider": "amazon_q",
                "name": "amazon_q",
                "role_arn": "arn:aws:iam::123456789012:role/example-role",
            }
        )
        before = current_model_metadata_with_size_context.get()

        scope_flat_rate_flow_models(CODE_REVIEW, USER)

        assert current_model_metadata_with_size_context.get() is before


class TestLogFlowModelFeature:
    """One line per workflow start with the client's feature next to the flow, which changes nothing."""

    FLOW_MODEL_FEATURE = "Flow model feature"

    def _logged(self, logs):
        return [entry for entry in logs if entry["event"] == self.FLOW_MODEL_FEATURE]

    @pytest.mark.asyncio
    async def test_logs_the_client_feature_and_the_model_that_passed(self):
        await _resolve_header(
            {
                "provider": "gitlab",
                "identifier": FABLE,
                "feature_setting": "duo_agent_platform",
            }
        )

        with capture_logs() as logs:
            log_flow_model_feature("security_review/v1", "security_review/v1")

        assert self._logged(logs) == [
            {
                "event": self.FLOW_MODEL_FEATURE,
                "log_level": "info",
                "flow": "security_review/v1",
                "workflow_definition": "security_review/v1",
                "client_feature_setting": "duo_agent_platform",
                "provider": "gitlab",
                "pinned": True,
                "requested_name": None,
                "requested_identifier": FABLE,
                "served_identifier": FABLE,
            }
        ]

    @pytest.mark.asyncio
    async def test_logs_a_resume_replay_without_a_feature(self):
        await _resolve_header({"provider": "gitlab", "name": FABLE})

        with capture_logs() as logs:
            log_flow_model_feature("developer/v1", "developer/v1")

        [entry] = self._logged(logs)
        assert entry["client_feature_setting"] is None
        assert entry["pinned"] is True
        assert entry["requested_name"] == FABLE

    @pytest.mark.asyncio
    async def test_logs_nothing_for_a_request_without_model_metadata(self):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})
        details = MagicMock()
        details.invocation_metadata = []
        await ModelMetadataInterceptor().intercept_service(AsyncMock(), details)

        with capture_logs() as logs:
            log_flow_model_feature("chat", "chat")

        assert self._logged(logs) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw",
        [None, [], {}, {"name": FABLE}],
        ids=["null", "list", "empty", "no-provider"],
    )
    async def test_logs_nothing_for_model_metadata_that_is_not_applied(self, raw):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})
        await _resolve_header(raw)

        with capture_logs() as logs:
            log_flow_model_feature("chat", "chat")

        assert self._logged(logs) == []

    @pytest.mark.asyncio
    async def test_logs_nothing_on_a_gateway_that_serves_custom_models(
        self, monkeypatch
    ):
        await _resolve_header({"provider": "gitlab", "identifier": FABLE})
        monkeypatch.setattr(get_config().custom_models, "enabled", True)

        with capture_logs() as logs:
            log_flow_model_feature("chat", "chat")

        assert self._logged(logs) == []

    @pytest.mark.asyncio
    async def test_changes_no_model(self):
        await _resolve_header(
            {
                "provider": "gitlab",
                "identifier": FABLE,
                "feature_setting": "duo_agent_platform",
            }
        )
        before = current_model_metadata_with_size_context.get()

        log_flow_model_feature(CODE_REVIEW, CODE_REVIEW)

        assert current_model_metadata_with_size_context.get() is before
