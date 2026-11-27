"""Guard for the auto mode classifier's LLM reviewer prompt."""

from unittest.mock import Mock

import pytest

from ai_gateway.config import ConfigModelLimits
from ai_gateway.prompts.base import jinja_env
from ai_gateway.prompts.registry import LocalPromptRegistry
from ai_gateway.response_schemas.registry import ResponseSchemaRegistry

PROMPT_ID = "tool_approval_review"
SCHEMA_ID = "tool_approval_review"
LATEST = "1.1.0"


@pytest.fixture(name="registry")
def registry_fixture():
    return LocalPromptRegistry.from_local_yaml(
        prompt_template_factories={},
        model_factories={},
        internal_event_client=Mock(),
        model_limits=ConfigModelLimits(),
    )


def _render_system(version: str = LATEST, **variables) -> str:
    return jinja_env.get_template(f"{PROMPT_ID}/system/{version}.jinja").render(
        today="2026-11-16", **variables
    )


@pytest.mark.parametrize("version", ["^1.0.0", "1.0.0"])
def test_the_registry_loads_the_prompt(registry, version):
    assert registry.get_required_variables(PROMPT_ID, version) >= {"goal", "today"}


def test_the_user_message_introduces_the_synthesized_goal():
    rendered = jinja_env.get_template(f"{PROMPT_ID}/user/{LATEST}.jinja").render(
        goal="Pending call: run_command"
    )

    assert rendered.strip() == (
        "Review the following pending tool call before it runs. Each section says who wrote it.\n\n"
        "Pending call: run_command"
    )


def test_the_reviewer_answers_with_the_response_schema():
    """The LLM backend reads the verdict from the schema tool's arguments, so the prompt must ask for that call."""
    schema = ResponseSchemaRegistry().get(SCHEMA_ID, "^1.0.0")
    rendered = _render_system()

    assert f"call `{schema.tool_title}` exactly once" in rendered
    for verdict in ("allow", "ask", "deny"):
        assert f"- {verdict}:" in rendered
    assert "Lead with" not in rendered


def test_the_reviewer_is_told_most_calls_need_no_reads():
    assert "Most calls can be judged from the call and the session alone" in (
        _render_system()
    )


@pytest.mark.parametrize("verdict", ["allow", "ask", "deny"])
def test_the_schema_accepts_each_verdict(verdict):
    schema = ResponseSchemaRegistry().get(SCHEMA_ID, "^1.0.0")

    assert schema(verdict=verdict, reason="why").to_output() == {
        "verdict": verdict,
        "reason": "why",
    }


@pytest.mark.parametrize(
    "args",
    [
        {"verdict": "Allow", "reason": "why"},
        {"verdict": "allow me to explain", "reason": "why"},
        {"verdict": "allow"},
        {"reason": "why"},
    ],
)
def test_the_schema_rejects_anything_else(args):
    """A malformed answer fails the reviewer's run, which the classifier turns into no verdict."""
    schema = ResponseSchemaRegistry().get(SCHEMA_ID, "^1.0.0")

    with pytest.raises(ValueError):
        schema(**args)


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
