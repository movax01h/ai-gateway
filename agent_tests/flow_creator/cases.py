"""The six-case test matrix for the Flow Creator agent.

The prompts are copied verbatim from the test matrix in
https://gitlab.com/gitlab-org/gitlab/-/work_items/604710 so that scores from
this suite are directly comparable to the manual end-to-end results recorded
against that work item.

Each case also declares which of the hard rules from
https://gitlab.com/gitlab-org/gitlab/-/work_items/604709 the case is expected
to exercise. ``test_hard_rules`` uses these flags to decide when a rule is
genuinely not applicable (and should be skipped rather than scored) versus
when the agent was expected to produce a construct and did not.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FlowCase:
    """A single prompt from the #604710 test matrix."""

    case_id: str
    title: str
    prompt: str

    # Expectations derived from the "Key requirements tested" column of the
    # matrix. These gate rule applicability, they are not assertions in
    # themselves.
    expects_gitlab_api_tools: bool = False
    expects_branch_context: bool = False
    expects_hitl: bool = False
    expects_supervisor: bool = False
    expects_multiple_agents: bool = False


CASES: tuple[FlowCase, ...] = (
    FlowCase(
        case_id="single_agent_summarizer",
        title="Single-agent, no HITL, reads files and summarizes",
        prompt=(
            "Create a flow that reads the README.md and any CHANGELOG file in a "
            "project, then produces a 3-4 sentence summary of what the project "
            "does and its most recent changes."
        ),
    ),
    FlowCase(
        case_id="gitlab_api_tools",
        title="Single-agent with GitLab API tools",
        prompt=(
            "Create a flow that lists all open issues with the label `bug` in a "
            "project, then posts a comment on the MR I specify summarizing how "
            "many open bugs there are and listing their titles."
        ),
        expects_gitlab_api_tools=True,
    ),
    FlowCase(
        case_id="two_agent_pipeline",
        title="Two-agent pipeline (generator to publisher)",
        prompt=(
            "Create a flow with two agents: the first reads all open issues and "
            "drafts release notes grouping them into New Features, Bug Fixes, "
            "and Breaking Changes. The second takes the draft and publishes it "
            "as a new Wiki page called Release Notes."
        ),
        expects_gitlab_api_tools=True,
        expects_multiple_agents=True,
    ),
    FlowCase(
        case_id="hitl_approval_gate",
        title="HITL approval gate with approve/modify/reject",
        prompt=(
            "Create a flow that analyzes the diff of a merge request and writes "
            "a code review summary. Before doing anything, it should pause and "
            "show me the summary for review. If I approve it posts it as a "
            "comment on the MR. If I ask for changes it revises and shows me "
            "again. If I reject it discards the output."
        ),
        expects_gitlab_api_tools=True,
        expects_hitl=True,
    ),
    FlowCase(
        case_id="branch_creation",
        title="Flow with branch creation (needs primary_branch)",
        prompt=(
            "Create a flow that creates a new feature branch from main, reads "
            "the failing test file I specify, attempts to fix the test, commits "
            "the change, and opens a draft MR targeting main."
        ),
        expects_gitlab_api_tools=True,
        expects_branch_context=True,
    ),
    FlowCase(
        case_id="supervisor_two_subagents",
        title="Supervisor with two sub-agents",
        prompt=(
            "Create a flow with a supervisor that coordinates a frontend "
            "developer and a backend developer. The supervisor reads an issue, "
            "plans the work, delegates frontend changes to one sub-agent and "
            "backend changes to the other, then summarizes what was implemented."
        ),
        expects_gitlab_api_tools=True,
        expects_supervisor=True,
        expects_multiple_agents=True,
    ),
)

CASES_BY_ID = {case.case_id: case for case in CASES}


# The agent's own Rule 8 ("Ask the user how the Custom Flow will be triggered")
# and its "Ask before designing" communication style mean the first turn is
# frequently a set of clarifying questions rather than YAML. These follow-ups
# answer those questions with fixed, realistic values so every case reaches a
# YAML output deterministically. They are replayed in order, and only while the
# agent has not yet produced a YAML document.
FOLLOW_UPS: tuple[str, ...] = (
    (
        "The flow is triggered manually by a user from the GitLab Duo Agent "
        "Platform UI. The project ID is 12345. It does not need to run on a "
        "schedule or react to webhooks. Please make reasonable assumptions for "
        "anything else and output the complete flow YAML now."
    ),
    (
        "Please output the complete flow YAML now, in a single yaml code block, "
        "with all required sections. Do not ask any more questions."
    ),
)
