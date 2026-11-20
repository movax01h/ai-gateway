"""Guard for the auto mode classifier's LLM reviewer prompt."""

from unittest.mock import Mock

import pytest

from ai_gateway.config import ConfigModelLimits
from ai_gateway.prompts.base import jinja_env
from ai_gateway.prompts.registry import LocalPromptRegistry

PROMPT_ID = "tool_approval_review"


@pytest.fixture(name="registry")
def registry_fixture():
    return LocalPromptRegistry.from_local_yaml(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=ConfigModelLimits(),
    )


def _render_system(**variables) -> str:
    return jinja_env.get_template(f"{PROMPT_ID}/system/1.0.0.jinja").render(
        today="2026-11-16", **variables
    )


def test_the_registry_loads_the_prompt(registry):
    assert registry.get_required_variables(PROMPT_ID, "^1.0.0") >= {"goal", "today"}


def test_the_user_message_is_the_synthesized_goal():
    rendered = jinja_env.get_template(f"{PROMPT_ID}/user/1.0.0.jinja").render(
        goal="Review this call"
    )

    assert rendered.strip() == "Review this call"


def test_the_reviewer_leads_with_one_of_the_three_verdicts():
    """The LLM backend parses a leading `allow`, `ask` or `deny`, so the prompt must ask for exactly that."""
    rendered = _render_system()

    assert "Lead with exactly one of these tokens" in rendered
    for verdict in ("allow", "ask", "deny"):
        assert f"- {verdict}:" in rendered


def test_the_project_renders_when_one_is_checked_out():
    rendered = _render_system(
        project_id=7,
        project_url="https://gitlab.example.com/group/project.git",
        primary_branch="main",
    )

    assert (
        "Checked-out project: https://gitlab.example.com/group/project.git "
        "(GitLab Project ID: 7, default branch: main)"
    ) in rendered


def test_the_project_is_omitted_when_none_is_checked_out():
    assert "Checked-out project" not in _render_system()


def test_the_workspace_rules_and_skills_never_reach_the_reviewer():
    """Anyone who can commit to the repo writes AGENTS.md, so it must not steer the safety reviewer."""
    rendered = _render_system(
        agents_dot_md="AGENTS.md body",
        workspace_agent_skills="<available_skills>skills</available_skills>",
    )

    assert "AGENTS.md body" not in rendered
    assert "<AGENTS.md>" not in rendered
    assert "<available_skills>" not in rendered
