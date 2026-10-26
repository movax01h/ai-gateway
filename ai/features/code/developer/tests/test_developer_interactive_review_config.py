"""Guard for the shipped developer/2.1.0-interactive flow config.

Only the non-obvious invariants are tested here; loading and dry-run compilation are covered generically in
test_registry.py and test_configs.py.
"""

import pytest

from ai_gateway.prompts.base import jinja_env
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig


def _review_agent():
    config = FlowConfig.from_yaml_config("developer", "2.1.0-interactive")
    return next(c for c in config.components if c["name"] == "review_agent")


def test_every_tool_call_is_approval_gated():
    """The security posture of this flow is that nothing the reviewer runs bypasses the developer.

    ``pre_approved_tools`` would reintroduce a flow-author-controlled bypass of the namespace
    admin's agent privileges, which is why it is being deprecated (#2744). It is not validated
    against ``toolset`` at load time and ``ToolApprovalRequestNode`` skips the prompt on a plain
    name match, so a single stray entry here would silently un-gate a tool.
    """
    review_agent = _review_agent()

    assert review_agent["require_tool_approval"] is True
    assert "pre_approved_tools" not in review_agent


def test_description_says_the_reviewer_gathers_its_own_context():
    """Observed: when the description understates the capability the supervisor refuses to delegate.

    It told the user the review agent "cannot run shell commands" and offered to run the command
    itself, which moves the call out of the sub-agent and defeats the point of delegating.
    """
    description = _review_agent()["description"].lower()

    assert "itself" in description and "diff" in description


def test_review_agent_cannot_write_anything():
    """It reports findings in the conversation; it must not be able to change or publish anything."""
    review_agent = _review_agent()

    assert set(review_agent["toolset"]) == {
        "run_command",
        "read_file",
        "read_files",
        "grep",
        "find_files",
        "list_dir",
    }


def test_review_agent_receives_the_workspace_context_the_supervisor_has():
    """Observed: the reviewer raised findings that a repository skill already settles.

    ``workspace_agent_skills`` and ``user_rule`` are flow-level context — ``_process_additional_context``
    puts them in state for every component — so the reviewer missed them only because it never bound
    them. Without them it reviews against generic conventions instead of the repository's own.
    """
    bound = {input_["as"] for input_ in _review_agent()["inputs"]}

    assert "workspace_agent_skills" in bound
    assert "agents_dot_md" in bound


def test_the_review_prompt_renders_the_workspace_context_it_binds():
    """A binding with no render site is inert: the input resolves and is then dropped silently."""
    template = jinja_env.get_template("local_code_review/system/1.0.0.jinja")

    rendered = template.render(
        today="2026-08-26",
        agents_dot_md="AGENTS.md body",
        workspace_agent_skills="<available_skills>skill catalogue</available_skills>",
    )

    assert "AGENTS.md body" in rendered
    assert "<available_skills>skill catalogue</available_skills>" in rendered


@pytest.mark.parametrize(
    "context",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"agents_dot_md": "", "workspace_agent_skills": ""}, id="empty"),
    ],
)
def test_the_review_prompt_stays_inert_when_the_client_sends_no_workspace_context(
    context,
):
    """Both inputs are optional, so the blocks must vanish rather than render empty envelopes.

    Omitted and empty are covered separately: the partials guard on truthiness so both behave the
    same today, but a guard rewritten as ``is defined`` would silently start emitting an empty
    envelope for the empty case.
    """
    rendered = jinja_env.get_template("local_code_review/system/1.0.0.jinja").render(
        today="August 27, 2026", **context
    )

    assert "<AGENTS.md>" not in rendered
    assert "<workspace_agent_skills>" not in rendered
