import pytest
from pydantic import ValidationError

from ai_gateway.model_selection.models import MAX_MODEL_RETRIES
from ai_gateway.prompts.config.base import PromptConfig, is_legacy_prompt


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


@pytest.mark.parametrize("tool_output_security", [True, False])
def test_the_format_renders_with_the_security_setting_it_is_given(
    tool_output_security,
):
    # The config's own `tool_output_security` is the caller's to pass: `Prompt._resolve` combines it with whether
    # tools are bound, which `tests/prompts/test_base.py` covers.
    config = PromptConfig(
        name="x",
        unit_primitive="duo_chat",
        prompt_template={"system": "instructions"},
    )

    prefix = config.prompt_format.resolve(
        tool_output_security=tool_output_security
    ).prefix

    assert (
        len(prefix) == 3
    ) is tool_output_security  # system, security block, history placeholder


@pytest.mark.parametrize(
    "declared,expected",
    [
        ({"prompt_template": {"system": "instructions"}}, True),
        ({"template": {"system": "instructions"}}, False),
    ],
    ids=["prompt_template", "template"],
)
def test_a_prompt_is_legacy_when_it_declares_the_old_roles(declared, expected):
    config = PromptConfig(name="x", unit_primitive="duo_chat", **declared)

    assert is_legacy_prompt(config) is expected
