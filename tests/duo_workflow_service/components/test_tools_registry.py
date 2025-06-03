import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.tools import BaseTool

from duo_workflow_service import tools
from duo_workflow_service.components.tools_registry import (
    _DEFAULT_TOOLS,
    NO_OP_TOOLS,
    Toolset,
    ToolsRegistry,
)
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
                "create_plan",
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
                "create_plan",
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "run_command",
                "handover_tool",
                "request_user_clarification_tool",
            },
        ),
        (
            ["read_only_gitlab"],
            {
                "create_plan",
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
                "get_repository_file",
                "list_epic_notes",
                "get_epic_note",
                "get_previous_workflow_context",
            },
        ),
        (
            ["read_write_gitlab"],
            {
                "create_plan",
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
                "get_repository_file",
                "list_epic_notes",
                "get_epic_note",
                "get_previous_workflow_context",
            },
        ),
        (
            ["use_git"],
            {
                "create_plan",
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
                "create_plan",
                "add_new_task",
                "remove_task",
                "update_task_description",
                "get_plan",
                "set_task_status",
                "read_file",
                "create_file_with_contents",
                "edit_file",
                "list_dir",
                "find_files",
                "grep",
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
def test_registry_initialization(tool_metadata, config, expected_tools_set):
    registry = ToolsRegistry(
        enabled_tools=config,
        preapproved_tools=[],
        tool_metadata=tool_metadata,
    )

    assert set(registry._enabled_tools.keys()) == expected_tools_set


def test_registry_initialization_initialises_tools_with_correct_attributes(
    tool_metadata,
):
    registry = ToolsRegistry(
        enabled_tools=[
            "run_commands",
            "use_git",
            "read_write_gitlab",
            "read_only_gitlab",
            "read_write_files",
        ],
        preapproved_tools=[],
        tool_metadata=tool_metadata,
    )
    expected_tools = {
        "create_plan": tools.CreatePlan(),
        "add_new_task": tools.AddNewTask(),
        "remove_task": tools.RemoveTask(),
        "update_task_description": tools.UpdateTaskDescription(),
        "get_plan": tools.GetPlan(),
        "set_task_status": tools.SetTaskStatus(),
        "run_command": tools.RunCommand(metadata=tool_metadata),
        "create_issue": tools.CreateIssue(metadata=tool_metadata),
        "list_issues": tools.ListIssues(metadata=tool_metadata),
        "get_issue": tools.GetIssue(metadata=tool_metadata),
        "update_issue": tools.UpdateIssue(metadata=tool_metadata),
        "get_job_logs": tools.GetLogsFromJob(metadata=tool_metadata),
        "get_merge_request": tools.GetMergeRequest(metadata=tool_metadata),
        "list_merge_request_diffs": tools.ListMergeRequestDiffs(metadata=tool_metadata),
        "create_merge_request_note": tools.CreateMergeRequestNote(
            metadata=tool_metadata
        ),
        "list_all_merge_request_notes": tools.ListAllMergeRequestNotes(
            metadata=tool_metadata
        ),
        "update_merge_request": tools.UpdateMergeRequest(metadata=tool_metadata),
        "get_pipeline_errors": tools.GetPipelineErrorsForMergeRequest(
            metadata=tool_metadata
        ),
        "get_project": tools.GetProject(metadata=tool_metadata),
        "gitlab_group_project_search": tools.GroupProjectSearch(metadata=tool_metadata),
        "gitlab_issue_search": tools.IssueSearch(metadata=tool_metadata),
        "gitlab_merge_request_search": tools.MergeRequestSearch(metadata=tool_metadata),
        "gitlab_milestone_search": tools.MilestoneSearch(metadata=tool_metadata),
        "gitlab__user_search": tools.UserSearch(metadata=tool_metadata),
        "gitlab_blob_search": tools.BlobSearch(metadata=tool_metadata),
        "gitlab_commit_search": tools.CommitSearch(metadata=tool_metadata),
        "gitlab_wiki_blob_search": tools.WikiBlobSearch(metadata=tool_metadata),
        "gitlab_note_search": tools.NoteSearch(metadata=tool_metadata),
        "read_file": tools.ReadFile(metadata=tool_metadata),
        "list_dir": tools.ListDir(metadata=tool_metadata),
        "create_file_with_contents": tools.WriteFile(metadata=tool_metadata),
        "edit_file": tools.EditFile(metadata=tool_metadata),
        "find_files": tools.FindFiles(metadata=tool_metadata),
        "grep": tools.Grep(metadata=tool_metadata),
        "mkdir": tools.Mkdir(metadata=tool_metadata),
        "run_git_command": tools.git.Command(metadata=tool_metadata),
        "handover_tool": tools.HandoverTool,
        "request_user_clarification_tool": tools.RequestUserClarificationTool,
        "get_epic": tools.GetEpic(metadata=tool_metadata),
        "list_epics": tools.ListEpics(metadata=tool_metadata),
        "create_epic": tools.CreateEpic(metadata=tool_metadata),
        "update_epic": tools.UpdateEpic(metadata=tool_metadata),
        "list_issue_notes": tools.ListIssueNotes(metadata=tool_metadata),
        "get_issue_note": tools.GetIssueNote(metadata=tool_metadata),
        "create_issue_note": tools.CreateIssueNote(metadata=tool_metadata),
        "create_merge_request": tools.CreateMergeRequest(metadata=tool_metadata),
        "get_repository_file": tools.GetRepositoryFile(metadata=tool_metadata),
        "list_epic_notes": tools.ListEpicNotes(metadata=tool_metadata),
        "get_epic_note": tools.GetEpicNote(metadata=tool_metadata),
        "get_previous_workflow_context": tools.GetWorkflowContext(
            metadata=tool_metadata
        ),
    }

    assert registry._enabled_tools == expected_tools


@pytest.mark.asyncio
async def test_registry_configuration(gl_http_client):
    workflow_config = {
        "id": "test_workflow",
        "agent_privileges_names": ["run_commands"],
    }
    extra_tool = MagicMock(spec=BaseTool)
    extra_tool.name = "extra_tool"

    registry = await ToolsRegistry.configure(
        workflow_config=workflow_config,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        gitlab_host="gitlab.example.com",
        additional_tools=[extra_tool],
    )

    # Verify configured tools based on privileges
    assert set(registry._enabled_tools.keys()) == {
        "create_plan",
        "add_new_task",
        "remove_task",
        "update_task_description",
        "get_plan",
        "set_task_status",
        "run_command",
        "handover_tool",
        "request_user_clarification_tool",
        "extra_tool",
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
def test_get_tool(tool_metadata, tool_name, expected_tool, config):
    registry = ToolsRegistry(
        enabled_tools=config,
        preapproved_tools=[],
        tool_metadata=tool_metadata,
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
def test_get_batch_tools(tool_metadata, requested_tools, expected_tools, config):
    registry = ToolsRegistry(
        enabled_tools=config,
        preapproved_tools=[],
        tool_metadata=tool_metadata,
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
def test_get_handlers(tool_metadata, requested_tools, expected_tools, config):
    registry = ToolsRegistry(
        enabled_tools=config,
        preapproved_tools=[],
        tool_metadata=tool_metadata,
    )

    assert [
        tool.__class__ for tool in registry.get_handlers(requested_tools)
    ] == expected_tools


def test_preapproved_tools_initialization(tool_metadata):
    registry = ToolsRegistry(
        enabled_tools=["read_write_files", "read_only_gitlab"],
        preapproved_tools=["read_write_files"],
        tool_metadata=tool_metadata,
    )

    # Default tools should always be in preapproved_tools
    default_tools = {
        "create_plan",
        "add_new_task",
        "remove_task",
        "update_task_description",
        "get_plan",
        "set_task_status",
        "handover_tool",
        "request_user_clarification_tool",
    }
    # Tools from read_write_files privilege should be in preapproved_tools
    read_write_tools = {
        "read_file",
        "create_file_with_contents",
        "edit_file",
        "list_dir",
        "find_files",
        "grep",
        "mkdir",
    }

    assert registry._preapproved_tool_names == default_tools.union(read_write_tools)


def test_approval_required(tool_metadata):
    registry = ToolsRegistry(
        enabled_tools=["read_write_files", "read_only_gitlab"],
        preapproved_tools=[
            "read_write_files"
        ],  # Only read_write_files tools are preapproved
        tool_metadata=tool_metadata,
    )

    # Tool is in preapproved list
    assert not registry.approval_required("read_file")  # from read_write_files
    assert not registry.approval_required("edit_file")  # from read_write_files

    # Tool is not in preapproved list
    assert registry.approval_required("list_issues")  # from read_only_gitlab
    assert registry.approval_required("get_issue")  # from read_only_gitlab

    # Tool that doesn't exist in enabled_tools but also not in preapproved list
    assert registry.approval_required("nonexistent_tool")


@pytest.mark.asyncio
async def test_registry_configuration_with_preapproved_tools(gl_http_client):
    workflow_config = {
        "id": "test_workflow",
        "agent_privileges_names": ["read_write_files", "run_commands"],
        "pre_approved_agent_privileges_names": ["read_write_files"],
    }

    registry = await ToolsRegistry.configure(
        workflow_config=workflow_config,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        gitlab_host="gitlab.example.com",
    )

    always_enabled_tools = set([tool_cls.tool_title for tool_cls in NO_OP_TOOLS])  # type: ignore
    always_enabled_tools.update([tool_cls().name for tool_cls in _DEFAULT_TOOLS])

    read_write_tools = {
        "read_file",
        "create_file_with_contents",
        "edit_file",
        "list_dir",
        "find_files",
        "grep",
        "mkdir",
    }
    expected_preapproved = always_enabled_tools.union(read_write_tools)

    assert registry._preapproved_tool_names == expected_preapproved
    assert registry.approval_required("run_command")
    assert not registry.approval_required("handover_tool")
    assert not registry.approval_required("add_new_task")
    assert not registry.approval_required("edit_file")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_config",
    [(None), ({"id": 123})],
    ids=["no_workflow", "no_agent_privileges_in_workflow"],
)
async def test_registry_configuration_error(gl_http_client, workflow_config):
    with pytest.raises(RuntimeError, match="Failed to find tools configuration"):
        await ToolsRegistry.configure(
            workflow_config=workflow_config,
            gl_http_client=gl_http_client,
            outbox=_outbox,
            inbox=_inbox,
            gitlab_host="gitlab.example.com",
        )


@pytest.mark.parametrize(
    "tool_names,expected_preapproved",
    [
        (
            ["read_file", "create_file_with_contents"],
            set(["read_file", "create_file_with_contents"]),
        ),
        (
            ["run_git_command"],
            set(),
        ),
        (
            ["read_file", "run_git_command"],
            {"read_file"},
        ),
        (
            ["nonexistent_tool"],  # Nonexistent tool should be filtered out
            set(),
        ),
    ],
    ids=[
        "with all tools being preapproved tools",
        "with all tools not being preapproved tools",
        "with mixed tools",
        "with nonexistent tool",
    ],
)
def test_toolset_method(tool_metadata, tool_names, expected_preapproved):
    registry = ToolsRegistry(
        enabled_tools=["read_write_files", "use_git"],
        preapproved_tools=["read_write_files"],
        tool_metadata=tool_metadata,
    )

    with patch("duo_workflow_service.components.tools_registry.Toolset") as MockToolset:
        mock_toolset = MagicMock(spec=Toolset)
        MockToolset.return_value = mock_toolset

        toolset = registry.toolset(tool_names)

        expected_all_tools = {
            tool_name: registry.get(tool_name)
            for tool_name in tool_names
            if registry.get(tool_name)
        }

        MockToolset.assert_called_once_with(
            pre_approved=expected_preapproved, all_tools=expected_all_tools
        )
        assert toolset == mock_toolset


@pytest.mark.parametrize(
    "feature_flag_value, should_include_commit_tools",
    [
        ("duo_workflow_commit_tools", True),
        ("", False),
    ],
)
@patch("duo_workflow_service.components.tools_registry.current_feature_flag_context")
def test_commit_tools_feature_flag(
    mock_feature_flags_context,
    feature_flag_value,
    should_include_commit_tools,
    tool_metadata,
):
    mock_feature_flags_context.get.return_value = feature_flag_value

    registry = ToolsRegistry(
        enabled_tools=["read_only_gitlab"],
        preapproved_tools=[],
        tool_metadata=tool_metadata,
    )

    assert ("get_commit" in registry._enabled_tools) == should_include_commit_tools
    assert ("list_commits" in registry._enabled_tools) == should_include_commit_tools
    assert ("get_commit_diff" in registry._enabled_tools) == should_include_commit_tools
    assert (
        "get_commit_comments" in registry._enabled_tools
    ) == should_include_commit_tools
