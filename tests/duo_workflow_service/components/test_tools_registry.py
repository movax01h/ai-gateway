import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service import tools
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


@pytest.fixture
def gl_http_client():
    return AsyncMock(spec=GitlabHttpClient)


_inbox = MagicMock(spec=asyncio.Queue)
_outbox = MagicMock(spec=asyncio.Queue)


@pytest.mark.parametrize(
    "config,expected_tools_set",
    [
        (
            {},
            {
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "handover_tool",
                "request_user_clarification_tool",
            },
        ),
        (
            ["run_commands"],
            {
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "run_command",
                "handover_tool",
                "run_read_only_git_command",
                "request_user_clarification_tool",
            },
        ),
        (
            ["read_only_gitlab"],
            {
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "list_issues",
                "get_issue",
                "get_job_logs",
                "get_merge_request",
                "list_merge_request_diffs",
                "list_all_merge_request_notes",
                "get_pipeline_errors",
                "get_project",
                "gitlab_group_project_search",
                "gitlab_issue_search",
                "gitlab_merge_request_search",
                "gitlab_milestone_search",
                "gitlab__user_search",
                "gitlab_blob_search",
                "gitlab_commit_search",
                "gitlab_wiki_blob_search",
                "gitlab_note_search",
                "handover_tool",
                "request_user_clarification_tool",
                "get_epic",
                "list_epics",
                "list_issue_notes",
                "get_issue_note",
            },
        ),
        (
            ["read_write_gitlab"],
            {
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "create_issue",
                "list_issues",
                "get_issue",
                "update_issue",
                "get_job_logs",
                "get_merge_request",
                "list_merge_request_diffs",
                "create_merge_request_note",
                "list_all_merge_request_notes",
                "update_merge_request",
                "get_pipeline_errors",
                "get_project",
                "gitlab_group_project_search",
                "gitlab_issue_search",
                "gitlab_merge_request_search",
                "gitlab_milestone_search",
                "gitlab__user_search",
                "gitlab_blob_search",
                "gitlab_commit_search",
                "gitlab_wiki_blob_search",
                "gitlab_note_search",
                "handover_tool",
                "request_user_clarification_tool",
                "get_epic",
                "list_epics",
                "create_epic",
                "update_epic",
                "create_issue_note",
                "create_merge_request",
                "list_issue_notes",
                "get_issue_note",
            },
        ),
        (
            ["use_git"],
            {
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "run_git_command",
                "handover_tool",
                "request_user_clarification_tool",
            },
        ),
        (
            ["read_write_files"],
            {
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "read_file",
                "create_file_with_contents",
                "edit_file",
                "ls_files",
                "find_files",
                "grep_files",
                "mkdir",
                "handover_tool",
                "request_user_clarification_tool",
            },
        ),
    ],
    ids=[
        "no_privileges",
        "run_command_privileges",
        "read_only_gitlab_privileges",
        "read_write_gitlab_privileges",
        "use_git_privileges",
        "read_write_files_privileges",
    ],
)
def test_registry_initialization(gl_http_client, config, expected_tools_set):
    registry = ToolsRegistry(
        outbox=_outbox,
        inbox=_inbox,
        gl_http_client=gl_http_client,
        tools_configuration=config,
        gitlab_host="gitlab.example.com",
    )

    assert set(registry._approved_tools.keys()) == expected_tools_set


def test_registry_initialization_initialises_tools_with_correct_attributes(
    gl_http_client,
):
    gitlab_host = "gitlab.example.com"
    registry = ToolsRegistry(
        outbox=_outbox,
        inbox=_inbox,
        gl_http_client=gl_http_client,
        tools_configuration=[
            "run_commands",
            "use_git",
            "read_write_gitlab",
            "read_only_gitlab",
            "read_write_files",
        ],
        gitlab_host=gitlab_host,
    )
    metadata = {
        "outbox": _outbox,
        "inbox": _inbox,
        "gitlab_client": gl_http_client,
        "gitlab_host": gitlab_host,
    }
    expected_tools = {
        "add_new_task": tools.AddNewTask(),
        "remove_task": tools.RemoveTask(),
        "update_task_description": tools.UpdateTaskDescription(),
        "get_plan": tools.GetPlan(),
        "set_task_status": tools.SetTaskStatus(),
        "run_command": tools.RunCommand(metadata=metadata),
        "create_issue": tools.CreateIssue(metadata=metadata),
        "list_issues": tools.ListIssues(metadata=metadata),
        "get_issue": tools.GetIssue(metadata=metadata),
        "update_issue": tools.UpdateIssue(metadata=metadata),
        "get_job_logs": tools.GetLogsFromJob(metadata=metadata),
        "get_merge_request": tools.GetMergeRequest(metadata=metadata),
        "list_merge_request_diffs": tools.ListMergeRequestDiffs(metadata=metadata),
        "create_merge_request_note": tools.CreateMergeRequestNote(metadata=metadata),
        "list_all_merge_request_notes": tools.ListAllMergeRequestNotes(
            metadata=metadata
        ),
        "update_merge_request": tools.UpdateMergeRequest(metadata=metadata),
        "get_pipeline_errors": tools.GetPipelineErrorsForMergeRequest(
            metadata=metadata
        ),
        "get_project": tools.GetProject(metadata=metadata),
        "gitlab_group_project_search": tools.GroupProjectSearch(metadata=metadata),
        "gitlab_issue_search": tools.IssueSearch(metadata=metadata),
        "gitlab_merge_request_search": tools.MergeRequestSearch(metadata=metadata),
        "gitlab_milestone_search": tools.MilestoneSearch(metadata=metadata),
        "gitlab__user_search": tools.UserSearch(metadata=metadata),
        "gitlab_blob_search": tools.BlobSearch(metadata=metadata),
        "gitlab_commit_search": tools.CommitSearch(metadata=metadata),
        "gitlab_wiki_blob_search": tools.WikiBlobSearch(metadata=metadata),
        "gitlab_note_search": tools.NoteSearch(metadata=metadata),
        "read_file": tools.ReadFile(metadata=metadata),
        "ls_files": tools.LsFiles(metadata=metadata),
        "create_file_with_contents": tools.WriteFile(metadata=metadata),
        "edit_file": tools.EditFile(metadata=metadata),
        "find_files": tools.FindFiles(metadata=metadata),
        "grep_files": tools.Grep(metadata=metadata),
        "mkdir": tools.Mkdir(metadata=metadata),
        "run_read_only_git_command": tools.ReadOnlyGit(metadata=metadata),
        "run_git_command": tools.git.Command(metadata=metadata),
        "handover_tool": tools.HandoverTool,
        "request_user_clarification_tool": tools.RequestUserClarificationTool,
        "get_epic": tools.GetEpic(metadata=metadata),
        "list_epics": tools.ListEpics(metadata=metadata),
        "create_epic": tools.CreateEpic(metadata=metadata),
        "update_epic": tools.UpdateEpic(metadata=metadata),
        "list_issue_notes": tools.ListIssueNotes(metadata=metadata),
        "get_issue_note": tools.GetIssueNote(metadata=metadata),
        "create_issue_note": tools.CreateIssueNote(metadata=metadata),
        "create_merge_request": tools.CreateMergeRequest(metadata=metadata),
    }

    assert registry._approved_tools == expected_tools


@pytest.mark.asyncio
async def test_registry_configuration(gl_http_client):
    workflow_id = "test_workflow"
    config = {
        "agent_privileges_names": ["run_commands"],
    }

    gl_http_client.aget.return_value = config

    registry = await ToolsRegistry.configure(
        workflow_id=workflow_id,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        gitlab_host="gitlab.example.com",
    )

    gl_http_client.aget.assert_called_once_with(
        f"/api/v4/ai/duo_workflows/workflows/{workflow_id}"
    )

    # Verify configured tools based on privileges
    assert set(registry._approved_tools.keys()) == {
        "add_new_task",
        "remove_task",
        "update_task_description",
        "get_plan",
        "set_task_status",
        "run_command",
        "run_read_only_git_command",
        "handover_tool",
        "request_user_clarification_tool",
    }


@pytest.mark.parametrize(
    "tool_name,expected_tool,config",
    [
        (
            "read_file",
            tools.ReadFile,
            ["read_write_files"],
        ),
        (
            "read_file",
            None,
            ["read_only_gitlab"],
        ),
        (
            "nonexistent_tool",
            None,
            ["read_write_files"],
        ),
        ("handover_tool", tools.HandoverTool, {}),
    ],
    ids=["approved_tool", "not_approved_tool", "nonexistent_tool", "handover_tool"],
)
def test_get_tool(gl_http_client, tool_name, expected_tool, config):
    registry = ToolsRegistry(
        outbox=_outbox,
        inbox=_inbox,
        gl_http_client=gl_http_client,
        tools_configuration=config,
        gitlab_host="gitlab.example.com",
    )

    tool = registry.get(tool_name)
    assert tool == expected_tool or isinstance(tool, expected_tool)


@pytest.mark.parametrize(
    "requested_tools,expected_tools,config",
    [
        (
            ["read_file", "list_issues", "nonexistent_tool"],
            [tools.ListIssues],
            ["read_only_gitlab"],
        ),
        (["nonexistent_tool"], [], {}),
    ],
    ids=["multiple_tools", "no_tools"],
)
def test_get_batch_tools(gl_http_client, requested_tools, expected_tools, config):
    registry = ToolsRegistry(
        outbox=_outbox,
        inbox=_inbox,
        gl_http_client=gl_http_client,
        tools_configuration=config,
        gitlab_host="gitlab.example.com",
    )

    assert [
        tool.__class__ for tool in registry.get_batch(requested_tools)
    ] == expected_tools


@pytest.mark.parametrize(
    "requested_tools,expected_tools,config",
    [
        (
            ["read_file", "handover_tool"],
            [tools.ReadFile],
            ["read_write_files"],
        ),
        (["handover_tool"], [], {}),
    ],
    ids=["tools_and_noop_tools_mixed", "noop_tools_only"],
)
def test_get_handlers(gl_http_client, requested_tools, expected_tools, config):
    registry = ToolsRegistry(
        outbox=_outbox,
        inbox=_inbox,
        gl_http_client=gl_http_client,
        tools_configuration=config,
        gitlab_host="gitlab.example.com",
    )

    assert [
        tool.__class__ for tool in registry.get_handlers(requested_tools)
    ] == expected_tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [(None), ({"id": 123})],
    ids=["no_response", "no_agent_privileges_names_in_a_response"],
)
async def test_registry_configuration_error(gl_http_client, config):
    gl_http_client.aget.return_value = config

    with pytest.raises(RuntimeError, match="Failed to fetch tools configuration"):
        await ToolsRegistry.configure(
            workflow_id="test_workflow",
            gl_http_client=gl_http_client,
            outbox=_outbox,
            inbox=_inbox,
            gitlab_host="gitlab.example.com",
        )
