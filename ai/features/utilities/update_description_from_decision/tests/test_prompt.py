"""Tests for the update_description_from_decision prompt that GitLab Rails calls on the prompt routes."""

from unittest.mock import Mock

import pytest

from ai_gateway.prompts.base import jinja_env
from ai_gateway.prompts.feature_roots import discover_feature_prompts
from ai_gateway.prompts.registry import LocalPromptRegistry

_PROMPT_ID = "update_description_from_decision"
_USER_TEMPLATE = "update_description_from_decision/user/1.0.0.jinja"
_INPUTS = {
    "title": "Add CSV export",
    "description": "## Goal\n\nExport issues.\n\nOpen question: which format?",
    "question": "Which format should the export use?",
    "options": ["CSV", "JSON"],
    "answer": "CSV",
    "rationale": "Spreadsheets open it directly.",
}


@pytest.fixture(scope="module", autouse=True)
def _feature_prompts():
    discover_feature_prompts()


@pytest.fixture(name="registry")
def registry_fixture() -> LocalPromptRegistry:
    return LocalPromptRegistry(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=Mock(),
        custom_models_enabled=False,
    )


def _render(**overrides) -> str:
    return jinja_env.get_template(_USER_TEMPLATE).render(**{**_INPUTS, **overrides})


@pytest.mark.parametrize("prompt_id", [_PROMPT_ID, f"utilities/{_PROMPT_ID}"])
def test_runs_on_the_duo_agent_platform_models(registry, prompt_id):
    assert registry.models_from(prompt_id, "^1.0.0") == "duo_agent_platform"


def test_requires_the_inputs_that_rails_sends(registry):
    assert registry.get_required_variables(_PROMPT_ID, "^1.0.0") == set(_INPUTS)


def test_renders_the_inputs():
    rendered = _render()

    assert f"<description>\n{_INPUTS['description']}\n</description>" in rendered
    assert "<options>\n- CSV\n- JSON\n</options>" in rendered
    assert "<answer>\nCSV\n</answer>" in rendered
    assert "<rationale>\nSpreadsheets open it directly.\n</rationale>" in rendered


def test_renders_a_written_in_answer_without_none():
    rendered = _render(answer="", rationale=None)

    assert "<answer>\n\n</answer>" in rendered
    assert "<rationale>\n\n</rationale>" in rendered
    assert "None" not in rendered
