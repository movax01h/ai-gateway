# These tests cover a cross-module feature (paid credit models), not one module.
# pylint: disable=file-naming-for-tests
import pytest

from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.model_selection.model_selection_config import (
    ModelTagEntry,
)

PAID_MODEL = "paid_model"
FREE_MODEL = "free_model"


@pytest.fixture(name="config")
def config_fixture():
    return ModelSelectionConfig(default_models_override={})


@pytest.fixture(name="models")
def models_fixture(config, monkeypatch):
    base = next(iter(config.get_llm_definitions().values()))
    paid = base.model_copy(
        deep=True,
        update={"gitlab_identifier": PAID_MODEL, "requires_paid_credits": True},
    )
    paid.params.model = "claude-opus-5-5"
    free = base.model_copy(
        deep=True,
        update={"gitlab_identifier": FREE_MODEL, "requires_paid_credits": False},
    )
    free.params.model = "claude-sonnet-5"
    models = {PAID_MODEL: paid, FREE_MODEL: free}
    monkeypatch.setattr(config, "get_llm_definitions", lambda: models)
    return models


@pytest.mark.usefixtures("models")
@pytest.mark.parametrize(
    ("identifiers", "model_names", "paid"),
    [
        ([PAID_MODEL], [], True),
        ([], ["anthropic/claude-opus-5-5"], True),
        ([], ["claude-opus-5-5@20261001"], True),
        ([], ["bedrock/global.anthropic.claude-opus-5-5"], True),
        ([FREE_MODEL], ["claude-sonnet-5"], False),
        ([None], [None], False),
    ],
)
def test_paid_credit_models_for(
    config: ModelSelectionConfig,
    identifiers: list[str | None],
    model_names: list[str | None],
    paid: bool,
):
    matched = config.paid_credit_models_for(identifiers, model_names)

    assert (PAID_MODEL in matched) is paid
    assert bool(matched) is paid


def test_validate_accepts_shipped_config(config: ModelSelectionConfig):
    config.validate()


def _validate_large_tag(
    config: ModelSelectionConfig, tag_models: list[str]
) -> list[str]:
    unit_primitive_configs = list(config.get_unit_primitive_config())
    duo_developer = next(
        c for c in unit_primitive_configs if c.feature_setting == "duo_developer"
    )
    duo_developer.models_for_tags["large"] = ModelTagEntry(models=tag_models)

    return config._validate_paid_credit_models(
        unit_primitive_configs, config.get_llm_definitions()
    )


@pytest.mark.usefixtures("models")
def test_validate_rejects_tag_with_only_paid_models(config: ModelSelectionConfig):
    errors = _validate_large_tag(config, [PAID_MODEL])

    assert errors == [
        "Tag 'large' of 'duo_developer' needs a model that does not require "
        "paid credits"
    ]


@pytest.mark.usefixtures("models")
def test_validate_accepts_tag_with_a_free_model(config: ModelSelectionConfig):
    errors = _validate_large_tag(config, [PAID_MODEL, FREE_MODEL])

    assert errors == []


@pytest.mark.usefixtures("models")
def test_validate_rejects_free_model_named_like_a_paid_model(
    config: ModelSelectionConfig,
):
    """The runtime check would deny a free model whose name extends a paid one."""
    free_variant = config.get_llm_definitions()[FREE_MODEL].model_copy(deep=True)
    free_variant.gitlab_identifier = "claude_opus_5_5_mini"
    free_variant.params.model = "claude-opus-5-5-mini"
    models = {**config.get_llm_definitions(), "claude_opus_5_5_mini": free_variant}

    errors = config._validate_paid_credit_models(
        list(config.get_unit_primitive_config()), models
    )

    assert len(errors) == 1
    assert errors[0].startswith(
        "Model 'claude_opus_5_5_mini' does not require paid credits but its model "
        "name matches paid model(s) ["
    )
    assert f"'{PAID_MODEL}'" in errors[0]


@pytest.mark.usefixtures("models")
def test_validate_rejects_env_release_free_model_named_like_a_paid_model(
    config: ModelSelectionConfig, monkeypatch
):
    """The runtime check matches env-injected releases too, so validation must see them."""
    release = config.get_llm_definitions()[FREE_MODEL].model_copy(deep=True)
    release.gitlab_identifier = "claude_opus_5_5_mini"
    release.params.model = "claude-opus-5-5-mini"
    monkeypatch.setitem(config._env_llm_definitions, release.gitlab_identifier, release)

    errors = config._validate_paid_credit_models(
        list(config.get_unit_primitive_config()), config.get_llm_definitions()
    )

    assert len(errors) == 1
    assert errors[0].startswith(
        "Model 'claude_opus_5_5_mini' does not require paid credits but its model "
        "name matches paid model(s) ["
    )
