# These tests cover a YAML file, so there is no module to name the file after.
# pylint: disable=file-naming-for-tests
"""Guards on the ``bl_security`` feature setting shipped in unit_primitives.yml.

The monolith selects this feature setting for the BL security flow.
"""

import pytest

from ai_gateway.model_selection import ModelSelectionConfig

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
