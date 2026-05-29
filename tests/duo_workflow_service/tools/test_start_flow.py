# pylint: disable=too-many-lines
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.start_flow import (
    FLOW_IDENTIFIER_MAP,
    StartCodeReviewFlowInput,
    StartDeveloperFlowInput,
    StartFixPipelineFlowInput,
    StartFlow,
    StartFlowInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="project")
def project_fixture():
    return {"id": 42, "web_url": "https://gitlab.com/group/project"}


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock, project):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project,
    }


@pytest.fixture(name="metadata_no_project")
def metadata_no_project_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture(name="tool")
def tool_fixture(metadata):
    return StartFlow(metadata=metadata)


@pytest.fixture(name="tool_no_project")
def tool_no_project_fixture(metadata_no_project):
    return StartFlow(metadata=metadata_no_project)


# ---------------------------------------------------------------------------
# FLOW_IDENTIFIER_MAP
# ---------------------------------------------------------------------------


def test_flow_identifier_map_covers_all_flow_names():
    assert FLOW_IDENTIFIER_MAP == {
        "developer": "developer/v1",
        "fix_pipeline": "fix_pipeline/v1",
        "code_review": "code_review/v1",
    }


# ---------------------------------------------------------------------------
# Tool description checks
# ---------------------------------------------------------------------------


def test_description_does_not_contain_user_approval_sentence(tool):
    assert (
        "The user must approve this tool call before the flow starts"
        not in tool.description
    )


def test_description_mentions_async_progress(tool):
    assert "asynchronously" in tool.description
    assert "session URL" in tool.description


def test_description_emphasizes_delegation(tool):
    assert "prefer" in tool.description
    assert "delegating" in tool.description


def test_description_does_not_contain_versioned_identifiers(tool):
    assert "fix_pipeline/v1" not in tool.description
    assert "code_review/v1" not in tool.description
    assert "developer/v1" not in tool.description


# ---------------------------------------------------------------------------
# Per-workflow input model validation
# ---------------------------------------------------------------------------


def test_start_developer_flow_input_requires_goal():
    with pytest.raises(ValidationError):
        StartDeveloperFlowInput(name="developer")


def test_start_developer_flow_input_accepts_goal():
    inp = StartDeveloperFlowInput(name="developer", goal="implement the login feature")
    assert inp.goal == "implement the login feature"


def test_start_fix_pipeline_flow_input_requires_all_fields():
    with pytest.raises(ValidationError):
        StartFixPipelineFlowInput(name="fix_pipeline")


def test_start_fix_pipeline_flow_input_validates_fields():
    inp = StartFixPipelineFlowInput(
        name="fix_pipeline",
        pipeline_url="https://gitlab.com/group/project/-/pipelines/123",
        merge_request_url="https://gitlab.com/group/project/-/merge_requests/1",
        source_branch="feature-branch",
    )
    assert inp.pipeline_url is not None
    assert (
        inp.merge_request_url == "https://gitlab.com/group/project/-/merge_requests/1"
    )
    assert inp.source_branch == "feature-branch"


def test_start_code_review_flow_input_requires_merge_request_url():
    with pytest.raises(ValidationError):
        StartCodeReviewFlowInput(name="code_review")


def test_start_code_review_flow_input_accepts_valid_url():
    inp = StartCodeReviewFlowInput(
        name="code_review",
        merge_request_url="https://gitlab.com/group/project/-/merge_requests/42",
    )
    assert inp.merge_request_url == (
        "https://gitlab.com/group/project/-/merge_requests/42"
    )


# ---------------------------------------------------------------------------
# StartFlowInput (discriminated union wrapper) validation
# ---------------------------------------------------------------------------


def test_start_flow_input_discriminates_developer():
    inp = StartFlowInput(
        flow={"name": "developer", "goal": "implement the login feature"}
    )
    assert isinstance(inp.flow, StartDeveloperFlowInput)


def test_start_flow_input_discriminates_fix_pipeline():
    inp = StartFlowInput(
        flow={
            "name": "fix_pipeline",
            "pipeline_url": "https://gitlab.com/group/project/-/pipelines/123",
            "merge_request_url": "https://gitlab.com/group/project/-/merge_requests/1",
            "source_branch": "main",
        }
    )
    assert isinstance(inp.flow, StartFixPipelineFlowInput)


def test_start_flow_input_discriminates_code_review():
    inp = StartFlowInput(
        flow={
            "name": "code_review",
            "merge_request_url": "https://gitlab.com/group/project/-/merge_requests/42",
        }
    )
    assert isinstance(inp.flow, StartCodeReviewFlowInput)


def test_start_flow_input_rejects_unknown_flow_name():
    with pytest.raises(ValidationError):
        StartFlowInput(flow={"name": "unknown_flow", "goal": "something"})


def test_start_flow_input_rejects_developer_without_goal():
    with pytest.raises(ValidationError):
        StartFlowInput(flow={"name": "developer"})


# ---------------------------------------------------------------------------
# _execute: fix_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_fix_pipeline_success(tool, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-123"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "fix_pipeline",
                "pipeline_url": "https://gitlab.com/group/project/-/pipelines/99",
                "merge_request_url": "https://gitlab.com/group/project/-/merge_requests/1",
                "source_branch": "feature-branch",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"
    assert data["workflow_id"] == "wf-123"
    assert data["flow_name"] == "fix_pipeline"
    assert (
        data["session_url"]
        == "https://gitlab.com/group/project/-/automate/agent-sessions/wf-123"
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["workflow_definition"] == "fix_pipeline/v1"
    assert "pipelines/99" in posted_body["goal"]
    assert posted_body["environment"] == "ambient"
    assert posted_body["start_workflow"] is True
    # project_id extracted from pipeline URL, not from self.project
    assert posted_body["project_id"] == "group/project"


@pytest.mark.asyncio
async def test_execute_code_review_cross_project(tool, gitlab_client_mock):
    """code_review with a URL from a different project uses that project."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-xp"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "code_review",
                "merge_request_url": (
                    "https://gitlab.com/other-group/other-project/-/merge_requests/7"
                ),
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["goal"] == "7"
    assert posted_body["project_id"] == "other-group/other-project"


@pytest.mark.asyncio
async def test_execute_fix_pipeline_cross_project(tool, gitlab_client_mock):
    """fix_pipeline with a URL from a different project uses that project."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-xp2"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "fix_pipeline",
                "pipeline_url": (
                    "https://gitlab.com/other-group/other-project/-/pipelines/55"
                ),
                "merge_request_url": (
                    "https://gitlab.com/other-group/other-project/-/merge_requests/3"
                ),
                "source_branch": "main",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["project_id"] == "other-group/other-project"


@pytest.mark.asyncio
async def test_execute_fix_pipeline_additional_context(tool, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-fp1"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "fix_pipeline",
                "pipeline_url": "https://gitlab.com/group/project/-/pipelines/99",
                "merge_request_url": "https://gitlab.com/group/project/-/merge_requests/1",
                "source_branch": "feature-branch",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["additional_context"] == [
        {
            "Category": "merge_request",
            "Content": json.dumps(
                {"url": "https://gitlab.com/group/project/-/merge_requests/1"}
            ),
        },
        {
            "Category": "pipeline",
            "Content": json.dumps({"source_branch": "feature-branch"}),
        },
    ]


# ---------------------------------------------------------------------------
# _execute: code_review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_code_review_success(tool, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-cr1"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "code_review",
                "merge_request_url": "https://gitlab.com/group/project/-/merge_requests/42",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["goal"] == "42"
    assert posted_body["workflow_definition"] == "code_review/v1"
    # project_id extracted from URL, not from self.project
    assert posted_body["project_id"] == "group/project"


@pytest.mark.asyncio
async def test_execute_string_body_response(tool, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=201,
            body=json.dumps({"id": "wf-str"}),
        )
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "code_review",
                "merge_request_url": "https://gitlab.com/group/project/-/merge_requests/42",
            }
        }
    )

    data = json.loads(result)
    assert data["workflow_id"] == "wf-str"


# ---------------------------------------------------------------------------
# _execute: developer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_developer_goal_only(tool, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-dev1"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Add a dark mode toggle to the settings page",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["goal"] == "Add a dark mode toggle to the settings page"
    assert posted_body["workflow_definition"] == "developer/v1"
    # Falls back to self.project when no project_url is provided
    assert posted_body["project_id"] == 42


@pytest.mark.asyncio
async def test_execute_developer_cross_project(tool, gitlab_client_mock):
    """Developer with project_url targets that project instead of self.project."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-dev-xp"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Implement feature X",
                "project_url": "https://gitlab.com/other-team/other-repo",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["project_id"] == "other-team/other-repo"
    assert posted_body["goal"] == "Implement feature X"


@pytest.mark.asyncio
async def test_execute_developer_does_not_add_additional_context(
    tool, gitlab_client_mock
):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-dev2"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Implement the feature described in the issue",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert "additional_context" not in posted_body


# ---------------------------------------------------------------------------
# _execute: no project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_without_project(tool_no_project, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-789"})
    )

    result = await tool_no_project.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Implement the feature described in the issue",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"
    assert data["workflow_id"] == "wf-789"
    assert data["session_url"] is None

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert "project_id" not in posted_body


# ---------------------------------------------------------------------------
# Invalid URL handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "merge_request_url",
    [
        "not-a-url",
        "https://other-host.com/group/project/-/merge_requests/1",
        "https://gitlab.com/group/project/-/pipelines/1",  # wrong resource type
        "https://gitlab.com/group/project/-/merge_requests/abc",  # non-numeric IID
        "https://gitlab.com/",  # no path
        "",
    ],
)
async def test_execute_code_review_invalid_merge_request_url(tool, merge_request_url):
    with pytest.raises(ToolException, match="Could not parse merge request URL"):
        await tool._execute(
            flow=StartCodeReviewFlowInput(
                name="code_review",
                merge_request_url=merge_request_url,
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pipeline_url",
    [
        "not-a-url",
        "https://other-host.com/group/project/-/pipelines/1",
        "https://gitlab.com/group/project/-/merge_requests/1",  # wrong resource type
        "https://gitlab.com/group/project/-/pipelines/abc",  # non-numeric IID
        "https://gitlab.com/",  # no path
        "",
    ],
)
async def test_execute_fix_pipeline_invalid_pipeline_url(tool, pipeline_url):
    with pytest.raises(ToolException, match="Could not parse pipeline URL"):
        await tool._execute(
            flow=StartFixPipelineFlowInput(
                name="fix_pipeline",
                pipeline_url=pipeline_url,
                merge_request_url="https://gitlab.com/group/project/-/merge_requests/1",
                source_branch="main",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project_url",
    [
        "not-a-url",
        "https://other-host.com/group/project",
        "https://gitlab.com/",  # no project path
    ],
)
async def test_execute_developer_invalid_project_url(tool, project_url):
    with pytest.raises(ToolException, match="Could not parse project URL"):
        await tool._execute(
            flow=StartDeveloperFlowInput(
                name="developer",
                goal="Implement feature X",
                project_url=project_url,
            )
        )


# ---------------------------------------------------------------------------
# _execute: HTTP errors and exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 403, 404, 422, 500])
async def test_execute_http_failure(tool, gitlab_client_mock, status_code):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=status_code, body="error body")
    )

    with pytest.raises(ToolException) as exc_info:
        await tool._execute(
            flow=StartFixPipelineFlowInput(
                name="fix_pipeline",
                pipeline_url="https://gitlab.com/group/project/-/pipelines/99",
                merge_request_url="https://gitlab.com/group/project/-/merge_requests/1",
                source_branch="feature-branch",
            ),
        )

    assert str(status_code) in str(exc_info.value)


@pytest.mark.asyncio
async def test_execute_exception(tool, gitlab_client_mock):
    gitlab_client_mock.apost = AsyncMock(side_effect=RuntimeError("network error"))

    with pytest.raises(RuntimeError, match="network error"):
        await tool._execute(
            flow=StartDeveloperFlowInput(
                name="developer",
                goal="Implement the feature",
            ),
        )


# ---------------------------------------------------------------------------
# format_display_message
# ---------------------------------------------------------------------------


def test_format_display_message_with_workflow_id_and_session_url(tool):
    args = StartFlowInput(
        flow=StartFixPipelineFlowInput(
            name="fix_pipeline",
            pipeline_url="https://example.com/pipelines/1",
            merge_request_url="https://example.com/merge_requests/1",
            source_branch="main",
        )
    )
    response = json.dumps(
        {
            "workflow_id": "wf-abc",
            "session_url": "https://gitlab.com/group/project/-/automate/agent-sessions/wf-abc",
        }
    )

    msg = tool.format_display_message(args, response)

    assert "fix_pipeline" in msg
    assert "wf-abc" in msg
    assert "View session" in msg


def test_format_display_message_with_workflow_id_no_session_url(tool):
    args = StartFlowInput(
        flow=StartCodeReviewFlowInput(
            name="code_review",
            merge_request_url="https://gitlab.com/group/project/-/merge_requests/99",
        )
    )
    response = json.dumps({"workflow_id": "wf-abc", "session_url": None})

    msg = tool.format_display_message(args, response)

    assert "code_review" in msg
    assert "wf-abc" in msg
    assert "View session" not in msg


@pytest.mark.parametrize(
    "response",
    [
        None,
        "not valid json{{{",
        json.dumps({}),  # missing workflow_id
    ],
)
def test_format_display_message_fallback_developer(tool, response):
    args = StartFlowInput(
        flow=StartDeveloperFlowInput(
            name="developer",
            goal="Add a dark mode toggle",
        )
    )

    msg = tool.format_display_message(args, response)

    assert "developer" in msg
    assert "Add a dark mode toggle" in msg


@pytest.mark.parametrize(
    "response",
    [
        None,
        "not valid json{{{",
        json.dumps({}),
    ],
)
def test_format_display_message_fallback_code_review(tool, response):
    args = StartFlowInput(
        flow=StartCodeReviewFlowInput(
            name="code_review",
            merge_request_url="https://gitlab.com/group/project/-/merge_requests/42",
        )
    )

    msg = tool.format_display_message(args, response)

    assert "code_review" in msg
    assert "merge_requests/42" in msg


@pytest.mark.parametrize(
    "response",
    [
        None,
        "not valid json{{{",
        json.dumps({}),
    ],
)
def test_format_display_message_fallback_fix_pipeline(tool, response):
    args = StartFlowInput(
        flow=StartFixPipelineFlowInput(
            name="fix_pipeline",
            pipeline_url="https://gitlab.com/group/project/-/pipelines/99",
            merge_request_url="https://gitlab.com/group/project/-/merge_requests/1",
            source_branch="main",
        )
    )

    msg = tool.format_display_message(args, response)

    assert "fix_pipeline" in msg
    assert "pipelines/99" in msg


def test_format_display_message_fallback_unknown_flow(tool):
    """The else branch is hit when flow_name is not one of the known values."""
    args = Mock()
    args.flow.model_dump.return_value = {"name": "unknown_flow", "extra": "data"}

    msg = tool.format_display_message(args, None)

    assert "unknown_flow" in msg


# ---------------------------------------------------------------------------
# _execute: defensive branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_raises_for_non_basemodel_flow(tool):
    """Line 110: the else branch raises ToolException for non-BaseModel input."""
    with pytest.raises(ToolException, match="Unexpected flow input type"):
        await tool._execute(flow="not-a-basemodel")


@pytest.mark.asyncio
async def test_execute_raises_for_unknown_flow_name(tool):
    """Line 115: raises ToolException when flow_name is not in FLOW_IDENTIFIER_MAP."""
    flow = Mock(spec=BaseModel)
    flow.model_dump.return_value = {"name": "unknown_flow"}
    with pytest.raises(ToolException, match="Unknown flow"):
        await tool._execute(flow=flow)


# ---------------------------------------------------------------------------
# _resolve_goal_project_and_linkable: defensive branch
# ---------------------------------------------------------------------------


def test_resolve_goal_project_and_linkable_raises_for_unknown_flow(tool):
    """Raises ToolException when flow_name is not developer/fix_pipeline/code_review."""
    with pytest.raises(ToolException, match="Unknown flow"):
        tool._resolve_goal_project_and_linkable(
            "unknown_flow", {"name": "unknown_flow"}
        )


# ---------------------------------------------------------------------------
# Explicit linkable linkage: issue_id / merge_request_id in payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_developer_with_issue_url(tool, gitlab_client_mock):
    """Developer flow with issue_url sends issue_id in payload."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-iss1"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Implement the feature described in the issue",
                "issue_url": "https://gitlab.com/group/project/-/issues/42",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"
    assert data["workflow_id"] == "wf-iss1"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["issue_id"] == 42
    assert "merge_request_id" not in posted_body
    assert posted_body["project_id"] == 42  # from self.project


@pytest.mark.asyncio
async def test_execute_developer_with_issue_url_and_project_url(
    tool, gitlab_client_mock
):
    """Developer flow with both issue_url and project_url sends both."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-iss2"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Fix the bug",
                "project_url": "https://gitlab.com/other-team/other-repo",
                "issue_url": "https://gitlab.com/other-team/other-repo/-/issues/7",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["project_id"] == "other-team/other-repo"
    assert posted_body["issue_id"] == 7


@pytest.mark.asyncio
async def test_execute_developer_cross_project_issue_uses_issue_project(
    tool, gitlab_client_mock
):
    """When project_url and issue_url differ, issue project wins."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-xp-iss"})
    )

    await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Fix the bug",
                "project_url": "https://gitlab.com/team-a/repo-a",
                "issue_url": "https://gitlab.com/team-b/repo-b/-/issues/42",
            }
        }
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    # Issue project takes precedence so Rails resolves the IID correctly
    assert posted_body["project_id"] == "team-b/repo-b"
    assert posted_body["issue_id"] == 42


@pytest.mark.asyncio
async def test_execute_developer_without_issue_url_omits_issue_id(
    tool, gitlab_client_mock
):
    """Developer flow without issue_url does not send issue_id."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-no-iss"})
    )

    await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Refactor the auth module",
            }
        }
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert "issue_id" not in posted_body
    assert "merge_request_id" not in posted_body


@pytest.mark.asyncio
async def test_execute_developer_with_work_item_url(tool, gitlab_client_mock):
    """Developer flow accepts work_items URLs via issue_url field."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-wi1"})
    )

    result = await tool.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Implement the feature",
                "issue_url": "https://gitlab.com/group/project/-/work_items/55",
            }
        }
    )

    data = json.loads(result)
    assert data["status"] == "started"

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["issue_id"] == 55


@pytest.mark.asyncio
async def test_execute_fix_pipeline_sends_merge_request_id(tool, gitlab_client_mock):
    """fix_pipeline flow sends merge_request_id in payload."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-fp-mr"})
    )

    await tool.arun(
        {
            "flow": {
                "name": "fix_pipeline",
                "pipeline_url": "https://gitlab.com/group/project/-/pipelines/99",
                "merge_request_url": (
                    "https://gitlab.com/group/project/-/merge_requests/5"
                ),
                "source_branch": "feature-branch",
            }
        }
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["merge_request_id"] == 5
    assert "issue_id" not in posted_body


@pytest.mark.asyncio
async def test_execute_code_review_sends_merge_request_id(tool, gitlab_client_mock):
    """code_review flow sends merge_request_id in payload."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-cr-mr"})
    )

    await tool.arun(
        {
            "flow": {
                "name": "code_review",
                "merge_request_url": (
                    "https://gitlab.com/group/project/-/merge_requests/42"
                ),
            }
        }
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["merge_request_id"] == 42
    assert "issue_id" not in posted_body


@pytest.mark.asyncio
async def test_execute_developer_issue_url_without_project_extracts_from_issue(
    tool_no_project, gitlab_client_mock
):
    """When no project context exists, project_id is extracted from issue_url."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-no-proj"})
    )

    await tool_no_project.arun(
        {
            "flow": {
                "name": "developer",
                "goal": "Fix the bug from the issue",
                "issue_url": "https://gitlab.com/group/project/-/issues/10",
            }
        }
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    assert posted_body["project_id"] == "group/project"
    assert posted_body["issue_id"] == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issue_url",
    [
        "not-a-url",
        "https://other-host.com/group/project/-/issues/1",
        "https://gitlab.com/",
    ],
)
async def test_execute_developer_invalid_issue_url(tool, issue_url):
    with pytest.raises(ToolException, match="Could not parse issue URL"):
        await tool._execute(
            flow=StartDeveloperFlowInput(
                name="developer",
                goal="Implement feature X",
                issue_url=issue_url,
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "merge_request_url",
    [
        "not-a-url",
        "https://other-host.com/group/project/-/merge_requests/1",
        "https://gitlab.com/group/project/-/pipelines/1",
        "https://gitlab.com/group/project/-/merge_requests/abc",
        "https://gitlab.com/",
        "",
    ],
)
async def test_execute_fix_pipeline_invalid_merge_request_url(tool, merge_request_url):
    """fix_pipeline with invalid merge_request_url raises ToolException."""
    with pytest.raises(ToolException, match="Could not parse merge request URL"):
        await tool._execute(
            flow=StartFixPipelineFlowInput(
                name="fix_pipeline",
                pipeline_url="https://gitlab.com/group/project/-/pipelines/99",
                merge_request_url=merge_request_url,
                source_branch="main",
            )
        )


@pytest.mark.asyncio
async def test_execute_fix_pipeline_cross_project_mr_uses_mr_project(
    tool, gitlab_client_mock
):
    """When pipeline and MR belong to different projects, MR project wins."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=201, body={"id": "wf-xp-mr"})
    )

    await tool.arun(
        {
            "flow": {
                "name": "fix_pipeline",
                "pipeline_url": ("https://gitlab.com/team-a/repo-a/-/pipelines/55"),
                "merge_request_url": (
                    "https://gitlab.com/team-b/repo-b/-/merge_requests/3"
                ),
                "source_branch": "main",
            }
        }
    )

    posted_body = json.loads(gitlab_client_mock.apost.call_args.kwargs["body"])
    # MR project takes precedence so Rails resolves the IID correctly
    assert posted_body["project_id"] == "team-b/repo-b"
    assert posted_body["merge_request_id"] == 3
