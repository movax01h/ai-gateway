# These tests cover a cross-module feature (paid credit models), not one module.
# pylint: disable=file-naming-for-tests
from unittest.mock import patch

import pytest
from structlog.testing import capture_logs

from ai_gateway.config import get_config
from ai_gateway.model_metadata import ModelMetadata, build_model_metadata_by_tag
from ai_gateway.model_selection import (
    ModelSelectionConfig,
    PaidCreditsModelAccessError,
    RestrictedModelAccessError,
    ensure_paid_credits_model_access,
)
from ai_gateway.model_selection.model_selection_config import (
    LLMDefinition,
    ModelTagEntry,
)
from lib.context.model import paid_credit_models_allowed_ctx

PAID_MODEL = "paid_model"
FREE_MODEL = "free_model"


def _metadata_for(llm_def: LLMDefinition, **_kwargs) -> ModelMetadata:
    return ModelMetadata(
        provider="gitlab", name=llm_def.gitlab_identifier, llm_definition=llm_def
    )


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


@pytest.fixture(name="claim")
def claim_fixture():
    return False


@pytest.fixture(name="mode")
def mode_fixture():
    return "enforce"


@pytest.fixture(autouse=True)
def paid_credit_models_gate(
    config: ModelSelectionConfig, claim: bool | None, mode: str, monkeypatch
):
    monkeypatch.setattr(
        get_config().model_selection, "paid_credit_models_gate_mode", mode
    )
    token = paid_credit_models_allowed_ctx.set(claim)
    with patch.object(ModelSelectionConfig, "instance", return_value=config):
        yield
    paid_credit_models_allowed_ctx.reset(token)


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


@pytest.mark.usefixtures("models")
def test_ensure_paid_credits_model_access_denies_trial_paid_model():
    with pytest.raises(PaidCreditsModelAccessError):
        ensure_paid_credits_model_access(identifiers=[PAID_MODEL])


@pytest.mark.usefixtures("models")
def test_ensure_paid_credits_model_access_logs_denial_in_log_mode(monkeypatch):
    monkeypatch.setattr(
        get_config().model_selection, "paid_credit_models_gate_mode", "log"
    )

    with capture_logs() as cap_logs:
        ensure_paid_credits_model_access(identifiers=[PAID_MODEL])

    assert [entry["event"] for entry in cap_logs] == ["Paid credits model denied"]
    assert cap_logs[0]["log_level"] == "warning"
    assert cap_logs[0]["mode"] == "log"
    assert PAID_MODEL in cap_logs[0]["paid_models"]


@pytest.mark.usefixtures("models")
@pytest.mark.parametrize(
    ("claim", "mode"),
    [
        (False, "off"),
        (None, "enforce"),
    ],
)
def test_ensure_paid_credits_model_access_does_not_log_when_ungated(claim, mode):
    with capture_logs() as cap_logs:
        ensure_paid_credits_model_access(identifiers=[PAID_MODEL])

    assert cap_logs == []


@pytest.mark.usefixtures("models")
@pytest.mark.parametrize(
    ("claim", "mode", "identifier"),
    [
        (False, "enforce", FREE_MODEL),
        (True, "enforce", PAID_MODEL),
    ],
)
def test_ensure_paid_credits_model_access_allows(identifier: str):
    ensure_paid_credits_model_access(identifiers=[identifier])


def test_paid_credits_denial_is_a_restricted_model_denial():
    """Existing restricted-model handlers report it as HTTP 403 / PERMISSION_DENIED."""
    with pytest.raises(RestrictedModelAccessError, match="purchased GitLab Credits"):
        ensure_paid_credits_model_access(models=["vertex_ai/claude-opus-5-5"])


@pytest.mark.parametrize(
    ("claim", "mode", "large_model"),
    [
        (False, "enforce", "kimi_k3_fireworks"),
        (False, "log", "claude_opus_5_5"),
        (True, "enforce", "claude_opus_5_5"),
        (None, "enforce", "claude_opus_5_5"),
    ],
)
def test_tag_models_exclude_paid_models_only_when_enforced(large_model: str):
    with (
        patch(
            "ai_gateway.model_metadata.random.choice",
            side_effect=lambda candidates: candidates[0],
        ),
        patch(
            "ai_gateway.model_metadata.resolve_provider_aware_metadata",
            side_effect=_metadata_for,
        ),
    ):
        by_tag = build_model_metadata_by_tag("duo_developer")

    large = by_tag["large"]
    assert isinstance(large, ModelMetadata)
    assert large.llm_definition.gitlab_identifier == large_model


@pytest.mark.usefixtures("models")
def test_tag_with_only_paid_models_falls_back_to_default(
    config: ModelSelectionConfig,
):
    duo_developer = config.get_unit_primitive_config_map()["duo_developer"]
    duo_developer.models_for_tags["large"] = ModelTagEntry(models=[PAID_MODEL])

    with patch(
        "ai_gateway.model_metadata.resolve_provider_aware_metadata",
        side_effect=_metadata_for,
    ):
        by_tag = build_model_metadata_by_tag("duo_developer")

    assert "large" not in by_tag


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
