# pylint: disable=file-naming-for-tests,redefined-outer-name,unsubscriptable-object
"""Tests for the developer_next and developer_unstable flow user prompt template rendering."""

import pytest
from jinja2 import Environment, StrictUndefined

from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
)

PROJECT_ID = 39903947
PROJECT_URL = (
    "https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist.git"
)
PROJECT_URL_NO_GIT = (
    "https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist"
)


def _load_user_prompt_template(flow_name: str) -> str:
    config = FlowConfig.from_yaml_config(flow_name)
    prompts = config.prompts
    assert prompts is not None
    raw = prompts[0]["prompt_template"]["user"]
    # Strip Jinja includes that reference files not available in the unit test context
    lines = [
        line
        for line in raw.splitlines()
        if not line.strip().startswith("{%- include")
        and not line.strip().startswith("{% include")
    ]
    return "\n".join(lines)


@pytest.fixture(scope="module", params=["developer_next", "developer_unstable"])
def user_prompt_template(request):
    return _load_user_prompt_template(request.param)


def render(template_str, **kwargs):
    defaults = {
        "project_id": PROJECT_ID,
        "project_url": PROJECT_URL,
        "goal": "",
    }
    defaults.update(kwargs)
    return (
        Environment(undefined=StrictUndefined)
        .from_string(template_str)
        .render(**defaults)
    )


class TestMentionTrigger:
    def _make_goal(self, input_text, context="Issue IID: 2030"):
        return f"Input: {input_text}\nContext: {{{context}}}"

    def test_mention_renders_mention_block(self, user_prompt_template):
        goal = self._make_goal("Please fix the bug described here.")
        result = render(user_prompt_template, goal=goal)
        assert "<mention>" in result
        assert "Please fix the bug described here." in result
        assert "</mention>" in result

    def test_mention_renders_resource_context_block(self, user_prompt_template):
        goal = self._make_goal("Do something.", context="Issue IID: 2030")
        result = render(user_prompt_template, goal=goal)
        assert "<resource_context>" in result
        assert "{Issue IID: 2030}" in result
        assert "</resource_context>" in result

    def test_mention_renders_project_id_in_api_path(self, user_prompt_template):
        goal = self._make_goal("Do something.")
        result = render(user_prompt_template, goal=goal)
        assert (
            f"/api/v4/projects/{PROJECT_ID}/{{resource_type}}/{{IID}}/notes" in result
        )

    def test_mention_renders_project_url_without_git_suffix(self, user_prompt_template):
        goal = self._make_goal("Do something.")
        result = render(user_prompt_template, goal=goal)
        assert f"{PROJECT_URL_NO_GIT}/-/{{resource_type}}/{{IID}}" in result
        assert ".git/-/" not in result

    def test_mention_project_url_git_in_path_not_stripped(self, user_prompt_template):
        """A .git segment in the middle of the URL must not be removed."""
        url_with_git_in_path = "https://gitlab.com/foo.git-stuff/repo.git"
        goal = self._make_goal("Do something.")
        result = render(
            user_prompt_template, goal=goal, project_url=url_with_git_in_path
        )
        assert "https://gitlab.com/foo.git-stuff/repo/-/" in result

    def test_mention_project_url_none_does_not_raise(self, user_prompt_template):
        """project_url is optional — None must not raise an AttributeError."""
        goal = self._make_goal("Do something.")
        result = render(user_prompt_template, goal=goal, project_url=None)
        assert "<mention>" in result

    def test_mention_input_not_in_resource_context(self, user_prompt_template):
        goal = self._make_goal("Please investigate this.", context="WorkItem IID: 99")
        result = render(user_prompt_template, goal=goal)
        assert "Please investigate this." not in result.split("<resource_context>")[1]

    def test_mention_context_not_in_mention_block(self, user_prompt_template):
        goal = self._make_goal("Do something.", context="MergeRequest IID: 42")
        result = render(user_prompt_template, goal=goal)
        assert "{MergeRequest IID: 42}" not in result.split("</mention>")[0]

    def test_mention_user_context_colon_in_input_is_not_split(
        self, user_prompt_template
    ):
        """User writes 'Context: ...' in their message — must not corrupt the split."""
        goal = self._make_goal("Foo bar\nContext: some user context text")
        result = render(user_prompt_template, goal=goal)
        mention_block = result.split("<mention>")[1].split("</mention>")[0]
        assert "Context: some user context text" in mention_block
        resource_context_block = result.split("<resource_context>")[1].split(
            "</resource_context>"
        )[0]
        assert "{Issue IID: 2030}" in resource_context_block

    def test_mention_goal_raw_not_present_outside_blocks(self, user_prompt_template):
        """The raw goal string should not appear verbatim outside the xml blocks."""
        goal = self._make_goal("Unique mention text XYZ123")
        result = render(user_prompt_template, goal=goal)
        # It should appear inside <mention>, not as the raw goal dump
        assert result.count("Unique mention text XYZ123") == 1

    def test_mention_guard_requires_exact_delimiter(self, user_prompt_template):
        """Goal with 'Context:' but without the exact delimiter must not trigger mention branch."""
        goal = "Input: something\nContext: no brace here"
        result = render(user_prompt_template, goal=goal)
        assert "<mention>" not in result


class TestMrReviewerTrigger:
    def test_mr_reviewer_branch_renders(self, user_prompt_template):
        goal = "https://gitlab.com/gitlab-org/gitlab/-/merge_requests/224998"
        result = render(user_prompt_template, goal=goal)
        assert "reviewer" in result.lower()
        assert goal in result

    def test_mr_reviewer_does_not_render_mention_blocks(self, user_prompt_template):
        goal = "https://gitlab.com/gitlab-org/gitlab/-/merge_requests/1"
        result = render(user_prompt_template, goal=goal)
        assert "<mention>" not in result
        assert "<resource_context>" not in result


class TestIssueTrigger:
    def test_issue_url_renders_solve_instruction(self, user_prompt_template):
        goal = "https://gitlab.com/gitlab-org/gitlab/-/issues/12345"
        result = render(user_prompt_template, goal=goal)
        assert "Solve the following GitLab issue" in result
        assert goal in result

    def test_issue_does_not_render_mention_blocks(self, user_prompt_template):
        goal = "https://gitlab.com/gitlab-org/gitlab/-/issues/12345"
        result = render(user_prompt_template, goal=goal)
        assert "<mention>" not in result
        assert "<resource_context>" not in result
