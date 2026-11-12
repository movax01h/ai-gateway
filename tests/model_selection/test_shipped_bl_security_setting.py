# These tests cover a YAML file, so there is no module to name the file after.
# pylint: disable=file-naming-for-tests
"""Guards on the ``bl_security`` feature setting shipped in unit_primitives.yml.

The monolith selects this feature setting for the BL security flow.
"""

from unittest.mock import patch

import pytest

from ai_gateway.model_selection import (
    ModelSelectionConfig,
    RestrictedModelAccessError,
    ensure_restricted_model_access,
)
from lib.context.model import restricted_access_ctx

_FEATURE_SETTING = "bl_security"


@pytest.fixture(name="selection_config", scope="module")
def selection_config_fixture() -> ModelSelectionConfig:
    # No overrides, so the assertions are about the shipped file and not a local `.env`.
    return ModelSelectionConfig(default_models_override={})


def test_bl_security_has_its_own_feature_setting(selection_config):
    assert _FEATURE_SETTING in selection_config.get_resolved_unit_primitive_config_map()


def test_every_default_model_is_selectable(selection_config):
    setting = selection_config.get_resolved_unit_primitive_config_map()[
        _FEATURE_SETTING
    ]
    assert setting.default_model_identifiers
    assert set(setting.default_model_identifiers) <= set(setting.selectable_models)


def test_the_default_model_sends_no_sampling_params(selection_config):
    # The default model rejects non-default sampling params, so carrying one would
    # fail every call.
    params = selection_config.get_model_for_feature(_FEATURE_SETTING).params
    assert getattr(params, "temperature", None) is None
    assert getattr(params, "top_p", None) is None


_MYTHOS_VERSIONS = ["5_1", "5_2"]


def _mythos_names(version: str) -> list[tuple[list[str], list[str]]]:
    model = f"claude-mythos-{version.replace('_', '-')}"
    return [
        ([f"claude_mythos_{version}"], []),
        ([], [model]),
        ([], [f"anthropic/{model}"]),
        ([], [f"{model}-20261001"]),
        ([], [f"{model}-latest"]),
        ([], [f"us.anthropic.{model}-v1:0"]),
        ([], [f"global.anthropic.{model}"]),
        ([], [f"bedrock/anthropic.{model}"]),
        ([], [f"vertex_ai/{model}@20261001"]),
    ]


_MYTHOS_NAMES = [
    pytest.param(identifiers, models, id=f"{version}-{models or identifiers}")
    for version in _MYTHOS_VERSIONS
    for identifiers, models in _mythos_names(version)
]


@pytest.fixture(name="authorized_flow")
def authorized_flow_fixture(selection_config, request):
    token = restricted_access_ctx.set(request.param)
    with patch.object(ModelSelectionConfig, "instance", return_value=selection_config):
        yield request.param
    restricted_access_ctx.reset(token)


@pytest.mark.parametrize("version", _MYTHOS_VERSIONS)
def test_mythos_is_selectable_for_bl_security_but_not_a_default(
    selection_config, version
):
    mythos = f"claude_mythos_{version}"
    setting = selection_config.get_unit_primitive_config_map()[_FEATURE_SETTING]
    assert mythos in setting.selectable_models
    assert mythos not in setting.default_model_identifiers


@pytest.mark.parametrize("version", _MYTHOS_VERSIONS)
def test_mythos_is_offered_by_no_other_feature(selection_config, version):
    mythos = f"claude_mythos_{version}"
    offering = {
        config.feature_setting
        for config in selection_config.get_unit_primitive_config()
        if mythos
        in {
            *config.default_model_identifiers,
            *config.selectable_models,
            *config.beta_models,
            *(config.dev.selectable_models if config.dev else []),
            *(m for e in config.models_for_tags.values() for m in e.models),
        }
    }
    assert offering == {_FEATURE_SETTING}


@pytest.mark.parametrize(
    "authorized_flow", [None, "chat", "code_review"], indirect=True
)
@pytest.mark.parametrize(("identifiers", "models"), _MYTHOS_NAMES)
def test_mythos_is_denied_outside_bl_security(authorized_flow, identifiers, models):
    with pytest.raises(RestrictedModelAccessError):
        ensure_restricted_model_access(identifiers, models)


@pytest.mark.parametrize("authorized_flow", [_FEATURE_SETTING], indirect=True)
@pytest.mark.parametrize(("identifiers", "models"), _MYTHOS_NAMES)
def test_mythos_is_allowed_for_bl_security(authorized_flow, identifiers, models):
    ensure_restricted_model_access(identifiers, models)


@pytest.mark.parametrize("version", _MYTHOS_VERSIONS)
def test_mythos_is_not_served_on_the_anthropic_proxy(selection_config, version):
    model = f"claude-mythos-{version.replace('_', '-')}"
    assert model not in selection_config.get_proxy_models_for_provider("anthropic")


def test_mythos_5_2_is_shaped_like_mythos_5_1(selection_config):
    old = selection_config.get_llm_definitions()["claude_mythos_5_1"]
    new = selection_config.get_llm_definitions()["claude_mythos_5_2"]
    assert new.params.model == "claude-mythos-5-2"
    assert new.params.max_tokens == old.params.max_tokens
    assert new.model_class_provider == old.model_class_provider
    assert new.max_context_tokens == old.max_context_tokens
    assert new.proxy_provider is None
    assert new.requires_paid_credits is True
    assert not getattr(new.params, "temperature", None)
    assert getattr(new.params, "fallbacks", None) is None


@pytest.mark.parametrize("version", _MYTHOS_VERSIONS)
def test_mythos_is_restricted_to_bl_security_in_the_shipped_restrictions(
    selection_config, version
):
    identifier = f"claude_mythos_{version}"
    assert selection_config.restricted_flows_for([identifier]) == {_FEATURE_SETTING}
    assert selection_config.restricted_flows_for(
        models=[f"claude-mythos-{version.replace('_', '-')}"]
    ) == {_FEATURE_SETTING}


def test_mythos_restrictions_do_not_cover_unrelated_models(selection_config):
    assert selection_config.restricted_flows_for(models=["claude-mythos-5-10"]) is None
    assert selection_config.restricted_flows_for(models=["claude-fable-5-2"]) is None


def test_mythos_5_2_drops_stale_thinking_blocks(selection_config):
    # Same preserved-thinking handling as claude_fable_5_2.
    params = selection_config.get_model("claude_mythos_5_2").params
    dumped = params.model_dump(exclude_none=True)
    assert dumped["thinking"] == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "drop_block"},
    }
    assert "thinking-binding-controls-2026-08-01" in dumped["betas"]
