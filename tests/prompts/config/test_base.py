import pytest
from pydantic import ValidationError

from ai_gateway.model_selection.models import MAX_MODEL_RETRIES
from ai_gateway.prompts.config.base import PromptConfig


def _prompt_config(max_retries: int) -> dict:
    return {
        "name": "test_prompt",
        "model": {"params": {"max_retries": max_retries}},
        "unit_primitive": "duo_chat",
        "prompt_template": {"user": "say hello"},
    }


@pytest.mark.parametrize("max_retries", [0, MAX_MODEL_RETRIES])
def test_prompt_config_accepts_max_retries_within_cap(max_retries):
    config = PromptConfig.model_validate(_prompt_config(max_retries))

    assert config.model.params.max_retries == max_retries


@pytest.mark.parametrize("max_retries", [-1, MAX_MODEL_RETRIES + 1])
def test_prompt_config_rejects_max_retries_outside_cap(max_retries):
    with pytest.raises(ValidationError, match="max_retries"):
        PromptConfig.model_validate(_prompt_config(max_retries))


@pytest.mark.parametrize(
    ("declared", "override", "has_security_block"),
    [
        (True, None, True),
        (False, None, False),
        (False, True, True),
        (True, False, False),
    ],
    ids=["own_setting_on", "own_setting_off", "override_on", "override_off"],
)
def test_resolve_takes_the_override_or_the_prompt_setting(
    declared, override, has_security_block
):
    config = PromptConfig(
        name="x",
        unit_primitive="duo_chat",
        prompt_template={"system": "instructions"},
        tool_output_security=declared,
    )

    prefix = config.resolve(tool_output_security=override).prefix

    assert (
        len(prefix) == 3
    ) is has_security_block  # system, security block, history placeholder
