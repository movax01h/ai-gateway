# These tests cover a cross-module feature (restricted models), not one module.
# pylint: disable=file-naming-for-tests
import json
import re
from unittest.mock import patch

import pytest

from ai_gateway.model_selection import (
    ModelSelectionConfig,
    RestrictedModelAccessError,
    ensure_restricted_model_access,
)
from ai_gateway.model_selection.model_selection_config import (
    DefaultModelEntry,
    ModelTagEntry,
)
from ai_gateway.model_selection.types import DevConfig
from ai_gateway.models.anthropic import KindAnthropicModel
from ai_gateway.models.litellm import KindGitLabModel, KindLiteLlmModel
from ai_gateway.proxy.clients.vertex_ai import _load_allowed_upstream_models
from lib.context.model import restricted_access_ctx
from tests.conftest import FAKE_RESTRICTED_MODEL

pytestmark = pytest.mark.usefixtures("fake_restricted_model")


@pytest.fixture(name="config")
def config_fixture():
    return ModelSelectionConfig(default_models_override={})


@pytest.fixture(name="authorized_flow")
def authorized_flow_fixture():
    return None


@pytest.fixture(autouse=True)
def restricted_access(config: ModelSelectionConfig, authorized_flow: str | None):
    token = restricted_access_ctx.set(authorized_flow)
    with patch.object(ModelSelectionConfig, "instance", return_value=config):
        yield
    restricted_access_ctx.reset(token)


@pytest.mark.parametrize(
    ("identifiers", "models"),
    [
        (["fake_restricted_model"], []),
        ([], ["claude-fake-restricted-1"]),
        ([], ["anthropic/claude-fake-restricted-1"]),
        ([], ["claude-fake-restricted-1@20261001"]),
        ([], ["us.anthropic.claude-fake-restricted-1-v1:0"]),
        ([], ["claude-fake-restricted-1-20261001"]),
        ([], ["vertex_ai/claude-fake-restricted-1:latest"]),
        ([], ["CLAUDE-FAKE-RESTRICTED-1"]),
        (["claude_sonnet_4_5_20250929"], ["claude-fake-restricted-1"]),
    ],
)
def test_restricted_flows_for_matches_identifier_and_model_strings(
    config: ModelSelectionConfig, identifiers: list[str], models: list[str]
):
    assert config.restricted_flows_for(identifiers, models) == {"bl_security"}


@pytest.mark.parametrize(
    ("identifiers", "models"),
    [
        (["claude_sonnet_4_5_20250929"], ["claude-sonnet-4-5-20250929"]),
        ([], ["not-claude-fake-restricted-1"]),
        ([], ["claude-fake-restricted-1-mini"]),
        ([], ["claude-fake-restricted-1.1"]),
        ([], ["anthropic/claude-fake-restricted-10"]),
        ([None], [None]),
        ([], []),
    ],
)
def test_restricted_flows_for_ignores_other_models(
    config: ModelSelectionConfig, identifiers: list, models: list
):
    assert config.restricted_flows_for(identifiers, models) is None


def test_restricted_flows_for_includes_env_releases_when_flag_is_off():
    released = {
        **FAKE_RESTRICTED_MODEL,
        "gitlab_identifier": "released_restricted_model",
        "params": {"model": "claude-released-restricted-1"},
    }
    config = ModelSelectionConfig(
        default_models_override={},
        model_releases=json.dumps({"models": [released]}),
    )

    assert config.restricted_flows_for(["released_restricted_model"]) == {"bl_security"}
    assert config.restricted_flows_for(models=["claude-released-restricted-1"]) == {
        "bl_security"
    }


@pytest.mark.parametrize("provider", ["anthropic", "openai", "vertex-ai"])
def test_proxy_models_exclude_restricted_models(
    config: ModelSelectionConfig, fake_restricted_model, provider: str
):
    fake_restricted_model.proxy_provider = provider
    proxy_models = config.get_proxy_models_for_provider(provider)

    assert proxy_models
    assert "claude-fake-restricted-1" not in proxy_models

    fake_restricted_model.restricted_to_flows = []
    assert "claude-fake-restricted-1" in config.get_proxy_models_for_provider(provider)


def test_vertex_proxy_hard_coded_models_exclude_restricted_models(
    fake_restricted_model,
):
    fake_restricted_model.params.model = "code-bison"

    allowed = _load_allowed_upstream_models()

    assert "code-bison" not in allowed
    assert "code-bison@002" not in allowed
    assert "text-bison" in allowed


def test_ensure_restricted_model_access_denies_without_authorization():
    with pytest.raises(RestrictedModelAccessError) as exc_info:
        ensure_restricted_model_access(identifiers=["fake_restricted_model"])

    # A ValueError would be swallowed into a silent fallback by existing callers.
    assert not isinstance(exc_info.value, ValueError)
    assert str(exc_info.value) == "Model not available for this flow"


@pytest.mark.parametrize("authorized_flow", ["developer"])
def test_ensure_restricted_model_access_denies_other_flow():
    with pytest.raises(RestrictedModelAccessError):
        ensure_restricted_model_access(models=["claude-fake-restricted-1"])


@pytest.mark.parametrize("authorized_flow", ["bl_security"])
def test_ensure_restricted_model_access_allows_authorized_flow():
    ensure_restricted_model_access(
        identifiers=["fake_restricted_model"], models=["claude-fake-restricted-1"]
    )


def test_ensure_restricted_model_access_allows_unrestricted_models():
    ensure_restricted_model_access(
        identifiers=["claude_sonnet_4_5_20250929"],
        models=["claude-sonnet-4-5-20250929"],
    )


LEGACY_CLIENT_MODELS = [
    *(m.value for m in KindAnthropicModel),
    *(m.value for m in KindLiteLlmModel),
    *(m.value for m in KindGitLabModel),
]


def test_legacy_client_models_are_never_restricted():
    """The legacy Anthropic/LiteLLM clients bypass Prompt, so no restricted model may be listed there."""
    config = ModelSelectionConfig.instance()

    assert (
        config.restricted_flows_for(LEGACY_CLIENT_MODELS, LEGACY_CLIENT_MODELS) is None
    )


def test_legacy_client_models_check_would_catch_a_restricted_entry():
    restricted_legacy = {
        **FAKE_RESTRICTED_MODEL,
        "gitlab_identifier": "restricted_legacy",
        "params": {"model": KindAnthropicModel.CLAUDE_HAIKU_4_5.value},
    }
    config = ModelSelectionConfig(
        default_models_override={},
        model_releases=json.dumps({"models": [restricted_legacy]}),
    )

    assert config.restricted_flows_for(LEGACY_CLIENT_MODELS, LEGACY_CLIENT_MODELS)


def _config_with(definition: dict) -> ModelSelectionConfig:
    return ModelSelectionConfig(
        default_models_override={},
        model_releases=json.dumps({"models": [definition]}),
    )


def test_gpt_style_sibling_models_are_not_restricted():
    config = _config_with(
        {
            **FAKE_RESTRICTED_MODEL,
            "gitlab_identifier": "restricted_gpt",
            "params": {"model": "gpt-6"},
        }
    )

    assert config.restricted_flows_for(models=["gpt-6"]) == {"bl_security"}
    assert config.restricted_flows_for(models=["us.openai.gpt-6-v1:0"]) == {
        "bl_security"
    }
    assert config.restricted_flows_for(models=["gpt-6-mini"]) is None
    assert config.restricted_flows_for(models=["gpt-6.1"]) is None


RESTRICTED_LIST_NAMES = [
    "default_models",
    "selectable_models",
    "beta_models",
    "dev.selectable_models",
    "models_for_tags",
]


def _offer_fake_restricted_model(unit_primitive_config, list_name: str) -> None:
    if list_name == "default_models":
        unit_primitive_config.default_models = [
            DefaultModelEntry(identifier="fake_restricted_model")
        ]
    elif list_name == "dev.selectable_models":
        unit_primitive_config.dev = DevConfig(
            selectable_models=["fake_restricted_model"]
        )
    elif list_name == "models_for_tags":
        unit_primitive_config.models_for_tags["large"] = ModelTagEntry(
            models=["fake_restricted_model"]
        )
    else:
        getattr(unit_primitive_config, list_name).append("fake_restricted_model")


@pytest.mark.parametrize("list_name", RESTRICTED_LIST_NAMES)
def test_validate_rejects_restricted_model_offered_by_another_feature(
    config: ModelSelectionConfig, list_name: str
):
    unit_primitive_config = next(
        c
        for c in config.get_unit_primitive_config()
        if c.feature_setting != "bl_security"
    )
    _offer_fake_restricted_model(unit_primitive_config, list_name)

    with pytest.raises(ValueError, match=rf"remove it from {re.escape(list_name)}"):
        config.validate()


@pytest.mark.parametrize("list_name", RESTRICTED_LIST_NAMES)
def test_validate_allows_restricted_model_in_its_own_flow_feature(
    config: ModelSelectionConfig, list_name: str
):
    unit_primitive_configs = list(config.get_unit_primitive_config())
    bl_security = next(
        c for c in unit_primitive_configs if c.feature_setting == "bl_security"
    )
    _offer_fake_restricted_model(bl_security, list_name)

    assert (
        config._validate_restricted_models(
            unit_primitive_configs, config.get_llm_definitions()
        )
        == []
    )


@pytest.mark.parametrize("flows", [["bl_security", ""], ["bl_security", "bl_security"]])
def test_validate_rejects_empty_or_duplicate_restricted_flows(
    config: ModelSelectionConfig, flows: list[str]
):
    definitions = config.get_llm_definitions()
    definitions["fake_restricted_model"].restricted_to_flows = flows

    with patch.object(
        ModelSelectionConfig, "get_llm_definitions", return_value=definitions
    ):
        with pytest.raises(ValueError, match="empty or duplicate entries"):
            config.validate()


def test_validate_accepts_shipped_config(config: ModelSelectionConfig):
    config.validate()


@pytest.mark.parametrize("flow", ["bl_security", "other_flow"])
def test_request_matching_two_restricted_models_needs_a_flow_both_allow(flow: str):
    """Two restricted definitions matching one request allow only their common flows."""
    config = _config_with(
        {
            **FAKE_RESTRICTED_MODEL,
            "gitlab_identifier": "other_restricted_model",
            "params": {"model": "claude-other-restricted-1"},
            "restricted_to_flows": ["other_flow"],
        }
    )
    request = {
        "identifiers": ["fake_restricted_model"],
        "models": ["claude-other-restricted-1"],
    }
    token = restricted_access_ctx.set(flow)
    try:
        with patch.object(ModelSelectionConfig, "instance", return_value=config):
            assert config.restricted_flows_for(**request) == frozenset()
            with pytest.raises(RestrictedModelAccessError):
                ensure_restricted_model_access(**request)
    finally:
        restricted_access_ctx.reset(token)
