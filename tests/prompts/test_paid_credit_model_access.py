# These tests cover a cross-module feature (paid credit models), not one module.
# pylint: disable=file-naming-for-tests
import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.config import get_config
from ai_gateway.model_selection import PaidCreditsModelAccessError
from ai_gateway.model_selection.models import BaseModelParams, ModelClassProvider
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config.base import ModelConfig, PromptConfig
from ai_gateway.prompts.typing import TypeModelFactory
from lib.context.model import paid_credit_models_allowed_ctx


@pytest.fixture(name="claim")
def claim_fixture():
    return False


@pytest.fixture(autouse=True)
def paid_credit_models_gate(claim: bool | None, monkeypatch):
    monkeypatch.setattr(
        get_config().model_selection, "paid_credit_models_gate_mode", "enforce"
    )
    token = paid_credit_models_allowed_ctx.set(claim)
    yield
    paid_credit_models_allowed_ctx.reset(token)


def _build_prompt(model_factory: TypeModelFactory, model_name: str) -> Prompt:
    config = PromptConfig(
        name="test_prompt",
        model=ModelConfig(params=BaseModelParams(model=model_name)),
        unit_primitive=GitLabUnitPrimitive.DUO_AGENT_PLATFORM,
        prompt_template={"system": "Hi", "user": "{{content}}"},
    )
    return Prompt(ModelClassProvider.LITE_LLM, model_factory, config)


@pytest.mark.parametrize(
    "model_name", ["claude-opus-5-5", "vertex_ai/claude-opus-5-5@20261001"]
)
def test_raw_paid_model_string_is_denied_to_trials_without_credits(
    model_factory: TypeModelFactory, model_name: str
):
    with pytest.raises(PaidCreditsModelAccessError):
        _build_prompt(model_factory, model_name)


@pytest.mark.parametrize(
    ("claim", "model_name"),
    [
        (False, "claude-sonnet-5"),
        (True, "claude-opus-5-5"),
        (None, "claude-opus-5-5"),
    ],
)
def test_prompt_is_built_when_paid_models_are_allowed_or_not_requested(
    model_factory: TypeModelFactory, model_name: str
):
    assert _build_prompt(model_factory, model_name).name == "test_prompt"
