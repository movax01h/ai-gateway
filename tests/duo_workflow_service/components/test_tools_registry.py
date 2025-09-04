import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.code_suggestions.language_server import LanguageServerVersion
from duo_workflow_service import tools
from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    _DEFAULT_TOOLS,
    NO_OP_TOOLS,
    Toolset,
    ToolsRegistry,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.tools.mcp_tools import (
    convert_mcp_tools_to_langchain_tool_classes,
)
from duo_workflow_service.tools.vulnerabilities.get_vulnerability_details import (
    GetVulnerabilityDetails,
)
from duo_workflow_service.tools.work_item import (
    GetWorkItem,
    GetWorkItemNotes,
    ListWorkItems,
)
from lib.feature_flags import current_feature_flag_context


@pytest.fixture(name="gl_http_client")
def gl_http_client_fixture():
    return AsyncMock(spec=GitlabHttpClient)


@pytest.fixture(name="mcp_tools")
def mcp_tools_fixture():
    mcp_tool_mock = MagicMock()
    mcp_tool_mock.name = "extra_tool"
    mcp_tool_mock.description = "extra tool description"
    mcp_tool_mock.inputSchema = ""

    return convert_mcp_tools_to_langchain_tool_classes(mcp_tools=[mcp_tool_mock])


_inbox = MagicMock(spec=asyncio.Queue)
_outbox = MagicMock(spec=asyncio.Queue)


@pytest.mark.parametrize(
    "config,expected_tools_set",
    [
        (
            [],
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
                "ci_linter",
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
                "gitlab_documentation_search",
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
                "list_repository_tree",
                "list_epic_notes",
                "get_previous_session_context",
                "list_vulnerabilities",
                "get_commit",
                "list_commits",
                "get_commit_diff",
                "get_commit_comments",
                "list_instance_audit_events",
                "list_group_audit_events",
                "list_project_audit_events",
                "get_current_user",
                "get_vulnerability_details",
            },
        ),
        (
            ["read_write_gitlab"],
            {
                "ci_linter",
                "create_plan",
                "add_new_task",
                "remove_task",
                "update_task_description",
                "update_vulnerability_severity",
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
                "gitlab_documentation_search",
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
                "list_repository_tree",
                "list_epic_notes",
                "get_previous_session_context",
                "list_vulnerabilities",
                "get_commit",
                "list_commits",
                "get_commit_diff",
                "get_commit_comments",
                "list_instance_audit_events",
                "list_group_audit_events",
                "list_project_audit_events",
                "create_commit",
                "dismiss_vulnerability",
                "confirm_vulnerability",
                "get_current_user",
                "create_work_item",
                "create_work_item_note",
                "link_vulnerability_to_issue",
                "get_vulnerability_details",
                "revert_to_detected_vulnerability",
                "create_vulnerability_issue",
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
                "read_files",
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
        mcp_tools=None,
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
        mcp_tools=None,
    )
    expected_tools = {
        "ci_linter": tools.CiLinter(metadata=tool_metadata),
        "create_plan": tools.CreatePlan(),
        "add_new_task": tools.AddNewTask(),
        "remove_task": tools.RemoveTask(),
        "update_task_description": tools.UpdateTaskDescription(),
        "update_vulnerability_severity": tools.UpdateVulnerabilitySeverity(
            metadata=tool_metadata
        ),
        "get_plan": tools.GetPlan(),
        "set_task_status": tools.SetTaskStatus(),
        "run_command": tools.RunCommand(metadata=tool_metadata),
        "create_issue": tools.CreateIssue(metadata=tool_metadata),
        "list_issues": tools.ListIssues(metadata=tool_metadata),
        "get_issue": tools.GetIssue(metadata=tool_metadata),
        "update_issue": tools.UpdateIssue(metadata=tool_metadata),
        "get_job_logs": tools.GetLogsFromJob(metadata=tool_metadata),
        "get_merge_request": tools.GetMergeRequest(metadata=tool_metadata),
        "gitlab_merge_request_search": tools.ListMergeRequest(metadata=tool_metadata),
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
        "gitlab_documentation_search": tools.DocumentationSearch(
            metadata=tool_metadata
        ),
        "gitlab_milestone_search": tools.MilestoneSearch(metadata=tool_metadata),
        "gitlab__user_search": tools.UserSearch(metadata=tool_metadata),
        "gitlab_blob_search": tools.BlobSearch(metadata=tool_metadata),
        "gitlab_commit_search": tools.CommitSearch(metadata=tool_metadata),
        "gitlab_wiki_blob_search": tools.WikiBlobSearch(metadata=tool_metadata),
        "gitlab_note_search": tools.NoteSearch(metadata=tool_metadata),
        "read_file": tools.ReadFile(metadata=tool_metadata),
        "read_files": tools.ReadFiles(metadata=tool_metadata),
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
        "list_repository_tree": tools.ListRepositoryTree(metadata=tool_metadata),
        "list_epic_notes": tools.ListEpicNotes(metadata=tool_metadata),
        "get_previous_session_context": tools.GetSessionContext(metadata=tool_metadata),
        "list_vulnerabilities": tools.ListVulnerabilities(metadata=tool_metadata),
        "get_commit": tools.GetCommit(metadata=tool_metadata),
        "list_commits": tools.ListCommits(metadata=tool_metadata),
        "get_commit_diff": tools.GetCommitDiff(metadata=tool_metadata),
        "get_commit_comments": tools.GetCommitComments(metadata=tool_metadata),
        "create_commit": tools.CreateCommit(metadata=tool_metadata),
        "dismiss_vulnerability": tools.DismissVulnerability(metadata=tool_metadata),
        "confirm_vulnerability": tools.ConfirmVulnerability(metadata=tool_metadata),
        "list_instance_audit_events": tools.ListInstanceAuditEvents(
            metadata=tool_metadata
        ),
        "list_group_audit_events": tools.ListGroupAuditEvents(metadata=tool_metadata),
        "list_project_audit_events": tools.ListProjectAuditEvents(
            metadata=tool_metadata
        ),
        "get_current_user": tools.GetCurrentUser(metadata=tool_metadata),
        "create_work_item": tools.CreateWorkItem(metadata=tool_metadata),
        "create_work_item_note": tools.CreateWorkItemNote(metadata=tool_metadata),
        "link_vulnerability_to_issue": tools.LinkVulnerabilityToIssue(
            metadata=tool_metadata
        ),
        "get_vulnerability_details": GetVulnerabilityDetails(metadata=tool_metadata),
        "revert_to_detected_vulnerability": tools.RevertToDetectedVulnerability(
            metadata=tool_metadata
        ),
        "create_vulnerability_issue": tools.CreateVulnerabilityIssue(
            metadata=tool_metadata
        ),
    }

    assert registry._enabled_tools == expected_tools


@pytest.mark.asyncio
async def test_registry_configuration(gl_http_client, mcp_tools, project_mock):
    workflow_config = {
        "id": "test_workflow",
        "agent_privileges_names": ["run_commands", "use_git", "run_mcp_tools"],
        "gitlab_host": "gitlab.example.com",
    }

    registry = await ToolsRegistry.configure(
        workflow_config=workflow_config,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        project=project_mock,
        mcp_tools=mcp_tools,
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
        "run_git_command",
        "handover_tool",
        "request_user_clarification_tool",
        "extra_tool",
    }
    assert registry.approval_required("extra_tool") == True
    assert registry._mcp_tool_names == ["extra_tool"]


@pytest.mark.parametrize(
    "tool_name,expected_tool,config",
    [
        (
            "read_files",
            tools.ReadFiles,
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
        ("handover_tool", tools.HandoverTool, []),
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
        (["nonexistent_tool"], [], []),
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
        (["handover_tool"], [], []),
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
        "read_files",
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
async def test_registry_configuration_with_preapproved_tools(
    gl_http_client, project_mock
):
    workflow_config = {
        "id": "test_workflow",
        "agent_privileges_names": ["read_write_files", "run_commands"],
        "pre_approved_agent_privileges_names": ["read_write_files"],
        "gitlab_host": "gitlab.example.com",
    }

    registry = await ToolsRegistry.configure(
        workflow_config=workflow_config,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        project=project_mock,
    )

    always_enabled_tools = set([tool_cls.tool_title for tool_cls in NO_OP_TOOLS])  # type: ignore
    always_enabled_tools.update([tool_cls().name for tool_cls in _DEFAULT_TOOLS])

    read_write_tools = {
        "read_file",
        "read_files",
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
async def test_registry_configuration_error(
    gl_http_client, workflow_config, project_mock
):
    with pytest.raises(RuntimeError, match="Failed to find tools configuration"):
        await ToolsRegistry.configure(
            workflow_config=workflow_config,
            gl_http_client=gl_http_client,
            outbox=_outbox,
            inbox=_inbox,
            project=project_mock,
        )


@pytest.mark.parametrize(
    "enabled_tools,user_can,available_tools,unavailable_tools",
    [
        (
            ["read_write_gitlab"],
            ["ask_issue"],
            {"get_issue", "get_project"},
            {"get_epic"},
        ),
        (
            ["read_write_gitlab"],
            ["ask_epic"],
            {"get_epic", "get_project"},
            {"get_issue"},
        ),
        (
            ["use_git"],
            ["ask_epic"],
            {"run_git_command"},
            {"get_epic"},
        ),
    ],
)
def test_available_tools_for_user(
    enabled_tools, user_can, available_tools, unavailable_tools
):
    user = MagicMock(spec=CloudConnectorUser)
    user.can = MagicMock(side_effect=lambda value: value in user_can)

    registry = ToolsRegistry(enabled_tools, [], {}, None, user)

    assert available_tools.issubset(set(registry._enabled_tools.keys()))
    assert not unavailable_tools.issubset(set(registry._enabled_tools.keys()))


@pytest.mark.parametrize(
    ("privileges", "tool_names", "expected_tool_names", "expected_preapproved"),
    [
        (
            ["read_write_files", "use_git", "nonexistent_privilege"],
            ["read_files", "create_file_with_contents"],
            ["read_files", "create_file_with_contents", "extra_tool"],
            set(["read_files", "create_file_with_contents"]),
        ),
        (
            ["read_write_files", "use_git", "run_mcp_tools"],
            ["read_files", "create_file_with_contents"],
            ["read_files", "create_file_with_contents", "extra_tool"],
            set(["read_files", "create_file_with_contents"]),
        ),
        (
            ["read_write_files", "use_git"],
            ["run_git_command"],
            ["run_git_command", "extra_tool"],
            set(),
        ),
        (
            ["read_write_files", "use_git"],
            ["read_files", "run_git_command"],
            ["read_files", "run_git_command", "extra_tool"],
            {"read_files"},
        ),
        (
            ["read_write_files", "use_git"],
            ["nonexistent_tool"],  # Nonexistent tool should be filtered out
            ["nonexistent_tool", "extra_tool"],
            set(),
        ),
    ],
    ids=[
        "with all tools being preapproved tools",
        "with mcp tools enabled",
        "with all tools not being preapproved tools",
        "with mixed tools",
        "with nonexistent tool",
    ],
)
def test_toolset_method(
    tool_metadata,
    privileges,
    tool_names,
    expected_tool_names,
    expected_preapproved,
    mcp_tools,
):
    registry = ToolsRegistry(
        enabled_tools=privileges,
        preapproved_tools=["read_write_files"],
        tool_metadata=tool_metadata,
        mcp_tools=mcp_tools,
    )

    with patch("duo_workflow_service.components.tools_registry.Toolset") as MockToolset:
        mock_toolset = MagicMock(spec=Toolset)
        MockToolset.return_value = mock_toolset

        toolset = registry.toolset(tool_names)

        expected_all_tools = {
            tool_name: registry.get(tool_name)
            for tool_name in expected_tool_names
            if registry.get(tool_name)
        }

        MockToolset.assert_called_once_with(
            pre_approved=expected_preapproved, all_tools=expected_all_tools
        )
        assert toolset == mock_toolset


@pytest.mark.parametrize(
    "feature_flag_value, should_include_work_item_tools",
    [
        ("duo_workflow_work_item_tools", True),
        ("", False),
    ],
)
def test_work_item_tools_feature_flag(
    feature_flag_value,
    should_include_work_item_tools,
    tool_metadata,
):
    current_feature_flag_context.set({feature_flag_value})

    registry = ToolsRegistry(
        enabled_tools=["read_only_gitlab"],
        preapproved_tools=[],
        tool_metadata=tool_metadata,
    )

    assert (
        "get_work_item" in registry._enabled_tools
    ) == should_include_work_item_tools
    assert (
        "list_work_items" in registry._enabled_tools
    ) == should_include_work_item_tools
    assert (
        "get_work_item_notes" in registry._enabled_tools
    ) == should_include_work_item_tools


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lsp_version,feature_flags,ff_disabled_tools",
    [
        ("0.0.1", "", {}),
        (
            "0.0.1",
            "duo_workflow_work_item_tools",
            {GetWorkItem, ListWorkItems, GetWorkItemNotes},
        ),
        ("7.42.999", "", {}),
        (
            "7.42.999",
            "duo_workflow_work_item_tools",
            {GetWorkItem, ListWorkItems, GetWorkItemNotes},
        ),
    ],
)
async def test_registry_configuration_with_restricted_language_server_client(
    gl_http_client, lsp_version, feature_flags, ff_disabled_tools, project_mock
):
    current_feature_flag_context.set(feature_flags)
    workflow_config = {
        "id": "test_workflow",
        "agent_privileges_names": list(_AGENT_PRIVILEGES.keys()),
        "pre_approved_agent_privileges_names": list(_AGENT_PRIVILEGES.keys()),
        "gitlab_host": "gitlab.example.com",
    }
    registry = await ToolsRegistry.configure(
        workflow_config=workflow_config,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        project=project_mock,
        language_server_version=LanguageServerVersion.from_string(lsp_version),
    )

    expected_tools = [
        *[tool_cls().name for tool_cls in _DEFAULT_TOOLS],
        *[tool_cls.tool_title for tool_cls in NO_OP_TOOLS],
        *[
            tool_cls().name
            for tool_cls in _AGENT_PRIVILEGES["read_only_gitlab"]
            if tool_cls not in ff_disabled_tools
        ],
    ]
    assert set(registry._enabled_tools.keys()).issubset(expected_tools)

    ignored_tools = [
        tool_cls().name
        for privilege in _AGENT_PRIVILEGES.keys()
        for tool_cls in _AGENT_PRIVILEGES[privilege]
        if privilege != "read_only_gitlab"
    ]
    assert set(registry._enabled_tools.keys()).intersection(ignored_tools) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lsp_version,feature_flags,ff_disabled_tools",
    [
        (None, "duo_workflow_work_item_tools", {}),
        (None, "", {GetWorkItem, ListWorkItems, GetWorkItemNotes}),
        ("7.43.0", "duo_workflow_work_item_tools", {}),
        ("7.43.0", "", {GetWorkItem, ListWorkItems, GetWorkItemNotes}),
        ("7.43.1", "duo_workflow_work_item_tools", {}),
        ("7.43.1", "", {GetWorkItem, ListWorkItems, GetWorkItemNotes}),
        ("8.0.0", "duo_workflow_work_item_tools", {}),
        ("8.0.0", "", {GetWorkItem, ListWorkItems, GetWorkItemNotes}),
    ],
)
async def test_registry_configuration_with_unrestricted_language_server_client(
    gl_http_client, lsp_version, feature_flags, ff_disabled_tools, project_mock
):
    current_feature_flag_context.set(feature_flags)
    workflow_config = {
        "id": "test_workflow",
        "agent_privileges_names": list(_AGENT_PRIVILEGES.keys()),
        "pre_approved_agent_privileges_names": list(_AGENT_PRIVILEGES.keys()),
        "gitlab_host": "gitlab.example.com",
    }
    registry = await ToolsRegistry.configure(
        workflow_config=workflow_config,
        gl_http_client=gl_http_client,
        outbox=_outbox,
        inbox=_inbox,
        project=project_mock,
        language_server_version=(
            LanguageServerVersion.from_string(lsp_version) if lsp_version else None
        ),
    )

    enabled_tools = set(registry._enabled_tools.keys())
    for privilege in _AGENT_PRIVILEGES.keys():
        for tool_cls in _AGENT_PRIVILEGES[privilege]:
            if tool_cls not in ff_disabled_tools:
                assert tool_cls().name in enabled_tools
