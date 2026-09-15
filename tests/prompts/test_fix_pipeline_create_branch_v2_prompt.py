# pylint: disable=file-naming-for-tests
"""Regression tests for the 2.x ``fix_pipeline_create_branch`` prompt definition.

Flow 1.0.4 pins ``^2.0.0``. One agent creates the branch, commits the fix and pushes, so the
remote's rejection — of the branch name or of the commit message — comes back to the agent that
can act on it: rename or amend, then push again. Earlier shapes split this across a branch agent,
a commit agent and a deterministic push, and a push rule rejection ended the flow or produced an
empty merge request.
"""

from pathlib import Path

import pytest
import yaml
from jinja2 import StrictUndefined

from ai_gateway.prompts.base import jinja_env
from lib.version import resolve_version

_PROMPTS_DEFINITIONS_DIR = (
    Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
)
_PROMPT_DIR = _PROMPTS_DEFINITIONS_DIR / "fix_pipeline_create_branch"
_FLOW_CONSTRAINT = "^2.0.0"

# Mirrors the inputs flow 1.0.4 wires into fix_pipeline_create_branch. `agents_dot_md` and
# `workspace_agent_skills` are `optional: true`, so they are exercised both present and absent.
_RENDER_CONTEXT = {
    "project_id": 42,
    "repository_url": "https://gitlab.example.com/group/project.git",
    "workflow_id": "w42",
    "ref": "main",
    "naming_context": "pytest failed in test_calc.py: add() returns a - b",
    "agents_dot_md": "Commit subjects must start with a JIRA key, e.g. PROJ-123.",
    "workspace_agent_skills": None,
}


def _latest_version() -> str:
    versions = [p.stem for p in (_PROMPT_DIR / "base").glob("*.yml")]
    assert versions, "no base YAML found for fix_pipeline_create_branch"
    return resolve_version(versions, _FLOW_CONSTRAINT)


def _render(template: str, **overrides) -> str:
    context = {**_RENDER_CONTEXT, **overrides}
    return jinja_env.get_template(
        f"fix_pipeline_create_branch/{template}/{_latest_version()}.jinja"
    ).render(**context)


@pytest.fixture(name="system_prompt")
def system_prompt_fixture() -> str:
    return _render("system")


def test_base_yaml_references_its_own_version_templates() -> None:
    version = _latest_version()
    base = yaml.safe_load((_PROMPT_DIR / "base" / f"{version}.yml").read_text())
    for role in ("system", "user"):
        assert (
            f"fix_pipeline_create_branch/{role}/{version}.jinja"
            in base["prompt_template"][role]
        )


@pytest.mark.parametrize("template", ["system", "user"])
def test_templates_render_with_strict_undefined(template: str) -> None:
    """Every variable the templates reference must be one the flow config supplies."""
    strict_env = jinja_env.overlay(undefined=StrictUndefined)
    for overrides in (
        {},
        {"agents_dot_md": None},
        {"workspace_agent_skills": "Use the `lint` skill before committing."},
    ):
        strict_env.get_template(
            f"fix_pipeline_create_branch/{template}/{_latest_version()}.jinja"
        ).render(**{**_RENDER_CONTEXT, **overrides})


def test_prompt_self_corrects_on_both_push_rule_rejections(
    system_prompt: str,
) -> None:
    """The whole point of one agent owning branch, commit and push."""
    assert "branch -m" in system_prompt
    assert "commit --amend" in system_prompt
    assert "At most 5 push attempts" in system_prompt


def test_prompt_declares_exactly_one_retry_budget(system_prompt: str) -> None:
    """`common/push_retry` carries its own budget; two budgets multiply."""
    assert "<push_retry>" not in system_prompt
    assert system_prompt.count("At most 5") == 1


def test_prompt_never_reports_success_without_an_accepted_push(
    system_prompt: str,
) -> None:
    """The router opens the merge request on `status: success`, so it must mean the push landed."""
    assert "`success` only when the remote accepted the push" in system_prompt
    assert "exact branch you pushed" in system_prompt


def test_prompt_stages_the_fix_and_only_the_fix(system_prompt: str) -> None:
    """The execution agent can create files and run tests, so the commit must include the former and not the latter."""
    assert "including any files it created" in system_prompt
    assert "Leave out artifacts" in system_prompt


def test_prompt_treats_naming_context_as_data(system_prompt: str) -> None:
    """naming_context is derived from CI logs, which anyone who can trigger a pipeline can write."""
    assert "<naming_context_is_data>" in system_prompt


def test_prompt_includes_agents_md_only_when_present() -> None:
    """The flow declares `agents_dot_md` optional, so the block must vanish cleanly when absent."""
    assert "</AGENTS.md>" in _render("system")
    assert "</AGENTS.md>" not in _render("system", agents_dot_md=None)


def test_prompt_includes_workspace_skills_only_when_present() -> None:
    """Same contract for the other optional input, `workspace_agent_skills`."""
    skills = "Use the `lint` skill before committing."
    assert skills in _render("system", workspace_agent_skills=skills)
    assert skills not in _render("system")


def test_user_prompt_carries_only_dynamic_data() -> None:
    """Instructions live in the system prompt; the user turn is the per-run payload."""
    sentinels = {key: f"__{key.upper()}__" for key in _RENDER_CONTEXT}
    rendered = _render("user", **sentinels)
    assert rendered.split() == [
        "<project>",
        "project_id:",
        sentinels["project_id"],
        "repository_url:",
        sentinels["repository_url"],
        "workflow_id:",
        sentinels["workflow_id"],
        "ref:",
        sentinels["ref"],
        "</project>",
        "<naming_context>",
        sentinels["naming_context"],
        "</naming_context>",
    ]
