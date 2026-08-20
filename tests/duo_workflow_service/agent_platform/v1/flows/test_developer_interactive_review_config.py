"""Guard for the shipped developer/2.1.0-interactive flow config.

Only the non-obvious invariants are tested here; loading and dry-run compilation are covered generically in
test_registry.py and test_configs.py.
"""

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
