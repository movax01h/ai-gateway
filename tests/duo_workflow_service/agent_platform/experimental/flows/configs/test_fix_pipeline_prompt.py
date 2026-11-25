# pylint: disable=file-naming-for-tests,unsubscriptable-object
"""Tests for the experimental fix_pipeline flow prompt templates.

The single-agent flow must cover the v1 fix_pipeline delivery contract, so these tests render the prompt through
LangChain's ChatPromptTemplate (as the AgentComponent does) and assert on the scenario-dependent content: MR,
branch and default-branch runs, the ``<fix_pipeline_source>`` value that AGENTS.md label rules key on, and the
messaging contract.
"""

from typing import Any

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ai_gateway.prompts.feature_roots import discover_feature_prompts
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)
from lib.prompts.utilities import prompt_template_to_messages

FLOW_NAME = "fix_pipeline"
PIPELINE_URL = "https://gitlab.example.com/group/project/-/pipelines/123"
MERGE_REQUEST_URL = "https://gitlab.example.com/group/project/-/merge_requests/5"
SESSION_URL = "https://gitlab.example.com/group/project/-/automate/agent-sessions/999"

# primary_branch is the project's default branch (context:project_default_branch), not the
# pipeline's own branch.
BASE_INPUTS: dict[str, Any] = {
    "pipeline_url": PIPELINE_URL,
    "merge_request_url": "",
    "merge_request_author_id": None,
    "source_branch": "feature-x",
    "primary_branch": "main",
    "project_id": 42,
    "project_url": "https://gitlab.example.com/group/project.git",
    "assignee_id": "7",
    "workflow_id": "999",
    "session_url": SESSION_URL,
    "today": "2026-09-09",
    "agents_dot_md": None,
    "workspace_agent_skills": None,
    "agent_user_environment": None,
    "os_information": None,
    "history": [],
}


@pytest.fixture(scope="module", name="chat_prompt")
def chat_prompt_fixture() -> ChatPromptTemplate:
    discover_feature_prompts()
    config = FlowConfig.from_yaml_config(FLOW_NAME, None)
    assert config.prompts is not None
    prompt_template = config.prompts[0].prompt_template
    assert prompt_template is not None
    messages = prompt_template_to_messages(prompt_template)
    return ChatPromptTemplate.from_messages(messages, template_format="jinja2")


def _render(chat_prompt: ChatPromptTemplate, **overrides) -> tuple[str, str]:
    """Return the rendered (system, user) message contents; system joins every system message."""
    messages = chat_prompt.invoke({**BASE_INPUTS, **overrides}).to_messages()
    system = "\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
    return system, str(messages[-1].content)


def _source_tag(value: str) -> str:
    return f"<fix_pipeline_source>{value}</fix_pipeline_source>"


class TestFixPipelineSource:
    """The run's source compares the pipeline branch with the project's default branch."""

    def test_merge_request_run(self, chat_prompt):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert _source_tag("merge_request") in system
        assert "A merge request is associated with this pipeline" in system

    def test_branch_run_without_merge_request(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert _source_tag("branch") in system
        assert "The failure occurred on a branch (`feature-x`)" in system

    def test_default_branch_run(self, chat_prompt):
        system, _ = _render(chat_prompt, source_branch="main")

        assert _source_tag("default_branch") in system
        assert "The failure occurred on the default branch (`main`)" in system

    @pytest.mark.parametrize(
        "overrides",
        [{"primary_branch": None}, {"primary_branch": None, "source_branch": None}],
        ids=["no default branch", "no branches"],
    )
    def test_unknown_branches_are_a_branch_run(self, chat_prompt, overrides):
        system, _ = _render(chat_prompt, **overrides)

        assert _source_tag("branch") in system

    def test_session_context_shows_project_default_branch(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "checked out from `feature-x`" in system
        assert "default branch: main" in system
        assert "default branch: feature-x" not in system


class TestDeliveryPolicy:
    def test_merge_request_mode_offers_inline_suggestions_and_comments(
        self, chat_prompt
    ):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert "Prefer delivering the fix as **inline code suggestions**" in system
        assert "`add_comment` decision" in system
        assert (
            "`permissions_failure` and `security_sensitive` findings included" in system
        )
        assert f"**Comment on the original MR** ({MERGE_REQUEST_URL}" in system
        assert (
            f"[original merge request]({MERGE_REQUEST_URL})'s diff, so it could not be delivered there"
            in system
        )
        assert "Without a merge request there is no place" not in system

    def test_no_merge_request_mode_has_no_comment_path(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert (
            "Without a merge request there is no place to leave a useful comment"
            in system
        )
        assert "Prefer delivering the fix as **inline code suggestions**" not in system
        assert "**Comment on the original MR**" not in system
        assert (
            "No open merge request was associated with the failing pipeline" in system
        )

    @pytest.mark.parametrize("merge_request_url", ["", MERGE_REQUEST_URL])
    def test_empty_diff_ends_without_remote_action(
        self, chat_prompt, merge_request_url
    ):
        system, _ = _render(chat_prompt, merge_request_url=merge_request_url)

        assert "leaves the working tree unchanged" in system
        assert "record `no_action`" in system

    def test_push_to_merge_request_branch_uses_explicit_refspec(self, chat_prompt):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert "`git push origin HEAD:refs/heads/feature-x`" in system

    def test_unknown_merge_request_branch_has_no_refspec(self, chat_prompt):
        system, _ = _render(
            chat_prompt, merge_request_url=MERGE_REQUEST_URL, source_branch=None
        )

        assert "HEAD:refs/heads/" not in system

    def test_agents_md_validation_directive_blocks_direct_suggestions(
        self, chat_prompt
    ):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert "not available when AGENTS.md requires validation" in system
        assert "no AGENTS.md directive that requires validation" in system


class TestOpeningNewMergeRequest:
    def test_branch_name_is_unique_per_run_and_self_correcting(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "`duo/fix/999-<short-description>`" in system
        assert "`duo/fix/` is a default, not a requirement" in system
        assert "rename with `git branch -m`" in system
        assert "At most 5 push attempts in total" in system
        assert "renaming cannot fix it" in system
        assert "with `git commit --amend -m`" in system

    def test_unrecoverable_push_failure_is_reported_on_the_merge_request(
        self, chat_prompt
    ):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert "report the failure on the merge request" in system

    def test_unrecoverable_push_failure_without_merge_request_is_not_reported(
        self, chat_prompt
    ):
        system, _ = _render(chat_prompt)

        assert "report the failure on the merge request" not in system

    def test_push_is_plain(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "`git push -u origin <branch>` (do not skip hooks)" in system
        assert "--no-verify" not in system

    def test_merge_request_metadata_matches_v1(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert f"title `[FixPipeline] {PIPELINE_URL}`" in system
        assert "glab mr create --yes --draft" in system
        assert "Draft: [FixPipeline]" not in system
        assert "target branch `feature-x`" in system
        assert f"`Fixes [failing pipeline]({PIPELINE_URL})`" in system

    def test_assignee_defaults_to_session_owner(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "`glab api users/7`" in system

    def test_assignee_prefers_merge_request_author(self, chat_prompt):
        system, _ = _render(
            chat_prompt,
            merge_request_url=MERGE_REQUEST_URL,
            merge_request_author_id="12",
        )

        assert "`glab api users/12`" in system
        assert "`glab api users/7`" not in system

    def test_unknown_assignee_is_not_looked_up(self, chat_prompt):
        system, _ = _render(chat_prompt, assignee_id=None)

        assert "glab api users/" not in system

    def test_labels_follow_agents_md_keyed_on_source(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "Pass `--label` only when AGENTS.md instructs you" in system
        assert "use the `<fix_pipeline_source>` value above" in system

    def test_new_merge_request_gets_explanatory_comment(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "**Comment on the new MR** once" in system
        assert "Hi @<assignee_username>" in system
        assert "**What to do next:**" in system


class TestMessagingFormat:
    def test_footer_links_docs_and_session(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert (
            "_Powered by [Duo Fix Pipeline](https://docs.gitlab.com/user/duo_agent_platform/flows/"
            f"foundational_flows/fix_pipeline/#give-feedback) flow_ - [session 999]({SESSION_URL})"
        ) in system

    def test_v1_comment_and_suggestion_lead_lines(self, chat_prompt):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert f"`Pipeline {PIPELINE_URL} could not be fixed automatically.`" in system
        assert "_Recommended fix for the failing job:" in system
        assert "never invent one" in system

    def test_mentions_are_forbidden_except_assignee_greeting(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "Never use @-mentions" in system
        assert "The only exception is the assignee greeting" in system


class TestScopeAndClassification:
    def test_shared_partials_are_included(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "<failure_categories>" in system
        assert "agents_rule_match" in system
        assert "<fix_types>" in system
        assert "<suggested_fix_type>comment_only</suggested_fix_type>" in system

    def test_zero_jobs_triage_and_job_filters(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "`pipeline_started_at` is null" in system
        assert "`downstream_pipeline_creation_failed`" in system
        assert "job-name filter in AGENTS.md" in system

    def test_ci_config_error_on_merge_request_is_never_just_a_comment(
        self, chat_prompt
    ):
        system, _ = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert "never record `no_action` for want of a failing job" in system
        assert "never settle for an advisory comment" in system

    def test_ci_config_error_without_merge_request_is_always_fixed(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "never record `no_action` for want of a failing job" in system
        assert "never settle for an advisory comment" not in system

    def test_child_pipeline_guidance(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert '`rules:if: $CI_PIPELINE_SOURCE == "parent_pipeline"`' in system


class TestLocalValidation:
    def test_known_os_is_named(self, chat_prompt):
        system, _ = _render(chat_prompt, os_information="Linux x86_64")

        assert "Local commands run on: Linux x86_64." in system

    def test_unknown_os_is_omitted(self, chat_prompt):
        system, _ = _render(chat_prompt)

        assert "Local commands run on:" not in system


class TestUserPrompt:
    def test_user_prompt_carries_pipeline_and_merge_request(self, chat_prompt):
        _, user = _render(chat_prompt, merge_request_url=MERGE_REQUEST_URL)

        assert f"<pipeline_url>{PIPELINE_URL}</pipeline_url>" in user
        assert f"<merge_request_url>{MERGE_REQUEST_URL}</merge_request_url>" in user

    def test_user_prompt_omits_missing_merge_request(self, chat_prompt):
        _, user = _render(chat_prompt)

        assert "<merge_request_url>" not in user
