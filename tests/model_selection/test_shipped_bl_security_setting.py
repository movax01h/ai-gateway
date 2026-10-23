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


_MYTHOS = "claude_mythos_5_1"


@pytest.fixture(name="authorized_flow")
def authorized_flow_fixture(selection_config, request):
    token = restricted_access_ctx.set(request.param)
    with patch.object(ModelSelectionConfig, "instance", return_value=selection_config):
        yield request.param
    restricted_access_ctx.reset(token)


def test_mythos_is_selectable_for_bl_security_but_not_a_default(selection_config):
    setting = selection_config.get_unit_primitive_config_map()[_FEATURE_SETTING]
    assert _MYTHOS in setting.selectable_models
    assert _MYTHOS not in setting.default_model_identifiers


def test_mythos_is_offered_by_no_other_feature(selection_config):
    offering = {
        config.feature_setting
        for config in selection_config.get_unit_primitive_config()
        if _MYTHOS
        in {
            *config.default_model_identifiers,
            *config.selectable_models,
            *config.beta_models,
            *(config.dev.selectable_models if config.dev else []),
            *(m for e in config.models_for_tags.values() for m in e.models),
        }
    }
    assert offering == {_FEATURE_SETTING}


_MYTHOS_NAMES = [
    ([_MYTHOS], []),
    ([], ["claude-mythos-5-1"]),
    ([], ["anthropic/claude-mythos-5-1"]),
    ([], ["claude-mythos-5-1-20261001"]),
    ([], ["us.anthropic.claude-mythos-5-1-v1:0"]),
    ([], ["vertex_ai/claude-mythos-5-1@20261001"]),
]


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


def test_mythos_is_not_served_on_the_anthropic_proxy(selection_config):
    assert "claude-mythos-5-1" not in selection_config.get_proxy_models_for_provider(
        "anthropic"
    )
