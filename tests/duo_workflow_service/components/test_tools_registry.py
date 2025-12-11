from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from langchain.tools import BaseTool
from pydantic import Field

from duo_workflow_service import tools
from duo_workflow_service.components.tools_registry import (
    _CAPABILITY_DEPENDENT_TOOLS,
    _DEFAULT_TOOLS,
    NO_OP_TOOLS,
    Toolset,
    ToolsRegistry,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.tools.branch import CreateBranch
from duo_workflow_service.tools.code_review import (
    BuildReviewMergeRequestContext,
    PostDuoCodeReview,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.findings.get_security_finding_details import (
    GetSecurityFindingDetails,
)
from duo_workflow_service.tools.findings.list_security_findings import (
    ListSecurityFindings,
)
from duo_workflow_service.tools.gitlab_api_generic import GitLabApiGet, GitLabGraphQL
from duo_workflow_service.tools.mcp_tools import (
    convert_mcp_tools_to_langchain_tool_classes,
)
from duo_workflow_service.tools.vulnerabilities.get_vulnerability_details import (
    GetVulnerabilityDetails,
)
from duo_workflow_service.tools.vulnerabilities.post_sast_fp_analysis_to_gitlab import (
    PostSastFpAnalysisToGitlab,
)
from duo_workflow_service.tools.vulnerabilities.post_secret_fp_analysis_to_gitlab import (
    PostSecretFpAnalysisToGitlab,
)
from duo_workflow_service.tools.wiki import GetWikiPage


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


_outbox = MagicMock(spec=Outbox)


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
                "get_pipeline_failing_jobs",
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
                "get_work_item",
                "list_work_items",
                "get_work_item_notes",
                "extract_lines_from_text",
                "run_glql_query",
                "build_review_merge_request_context",
                "get_security_finding_details",
                "list_security_findings",
                "get_wiki_page",
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
                "create_branch",
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
                "get_pipeline_failing_jobs",
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
                "link_vulnerability_to_merge_request",
                "get_vulnerability_details",
                "update_work_item",
                "revert_to_detected_vulnerability",
                "create_vulnerability_issue",
                "get_work_item",
                "list_work_items",
                "get_work_item_notes",
                "post_duo_code_review",
                "extract_lines_from_text",
                "run_glql_query",
                "post_sast_fp_analysis_to_gitlab",
                "post_secret_fp_analysis_to_gitlab",
                "build_review_merge_request_context",
                "get_security_finding_details",
                "list_security_findings",
                "get_wiki_page",
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
                "extract_lines_from_text",
                "handover_tool",
                "request_user_clarification_tool",
                "run_tests",
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
@patch("duo_workflow_service.components.tools_registry.is_feature_enabled")
def test_registry_initialization(
    mock_is_feature_enabled, tool_metadata, config, expected_tools_set
):
    mock_is_feature_enabled.return_value = False

    registry = ToolsRegistry(
        enabled_tools=config,
        preapproved_tools=[],
        tool_metadata=tool_metadata,
        mcp_tools=None,
    )

    assert set(registry._enabled_tools.keys()) == expected_tools_set


@patch("duo_workflow_service.components.tools_registry.is_feature_enabled")
def test_registry_initialization_initialises_tools_with_correct_attributes(
    mock_is_feature_enabled,
    tool_metadata,
):
    mock_is_feature_enabled.return_value = False

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
        "get_pipeline_failing_jobs": tools.GetPipelineFailingJobs(
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
        "link_vulnerability_to_merge_request": tools.LinkVulnerabilityToMergeRequest(
            metadata=tool_metadata
        ),
        "get_vulnerability_details": GetVulnerabilityDetails(metadata=tool_metadata),
        "update_work_item": tools.UpdateWorkItem(metadata=tool_metadata),
        "revert_to_detected_vulnerability": tools.RevertToDetectedVulnerability(
            metadata=tool_metadata
        ),
        "create_vulnerability_issue": tools.CreateVulnerabilityIssue(
            metadata=tool_metadata
        ),
        "get_work_item": tools.GetWorkItem(metadata=tool_metadata),
        "list_work_items": tools.ListWorkItems(metadata=tool_metadata),
        "get_work_item_notes": tools.GetWorkItemNotes(metadata=tool_metadata),
        "post_duo_code_review": PostDuoCodeReview(metadata=tool_metadata),
        "extract_lines_from_text": tools.ExtractLinesFromText(metadata=tool_metadata),
        "run_glql_query": tools.RunGLQLQuery(metadata=tool_metadata),
        "post_sast_fp_analysis_to_gitlab": PostSastFpAnalysisToGitlab(
            metadata=tool_metadata
        ),
        "post_secret_fp_analysis_to_gitlab": PostSecretFpAnalysisToGitlab(
            metadata=tool_metadata
        ),
        "run_tests": tools.RunTests(metadata=tool_metadata),
        "build_review_merge_request_context": BuildReviewMergeRequestContext(
            metadata=tool_metadata
        ),
        "get_security_finding_details": GetSecurityFindingDetails(
            metadata=tool_metadata
        ),
        "list_security_findings": ListSecurityFindings(metadata=tool_metadata),
        "get_wiki_page": GetWikiPage(metadata=tool_metadata),
        "create_branch": CreateBranch(metadata=tool_metadata),
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
    assert registry.approval_required("extra_tool") is True
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
        "extract_lines_from_text",
        "run_tests",
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
        project=project_mock,
    )

    always_enabled_tools = {tool_cls.tool_title for tool_cls in NO_OP_TOOLS}
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
        "extract_lines_from_text",
        "run_tests",
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
            project=project_mock,
        )


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

    with patch(
        "duo_workflow_service.components.tools_registry.Toolset"
    ) as mock_toolset_class:
        mock_toolset = MagicMock(spec=Toolset)
        mock_toolset_class.return_value = mock_toolset

        toolset = registry.toolset(tool_names)

        expected_all_tools = {
            tool_name: registry.get(tool_name)
            for tool_name in expected_tool_names
            if registry.get(tool_name)
        }

        mock_toolset_class.assert_called_once_with(
            pre_approved=expected_preapproved, all_tools=expected_all_tools
        )
        assert toolset == mock_toolset


# Tests for Generic GitLab API Tools


class TestGenericGitLabAPITools:
    """Tests for generic GitLab API tools integration with ToolsRegistry."""

    @patch("duo_workflow_service.components.tools_registry.is_feature_enabled")
    def test_generic_tools_available_with_feature_flag_enabled(
        self, mock_is_feature_enabled, tool_metadata
    ):
        """Test that generic tools are available when feature flag is enabled."""
        mock_is_feature_enabled.return_value = True

        registry = ToolsRegistry(
            enabled_tools=["read_only_gitlab"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        # Verify generic tools are present when feature flag is enabled
        assert "gitlab_api_get" in registry._enabled_tools
        assert "gitlab_graphql" in registry._enabled_tools
        assert isinstance(registry._enabled_tools["gitlab_api_get"], GitLabApiGet)
        assert isinstance(registry._enabled_tools["gitlab_graphql"], GitLabGraphQL)

    @patch("duo_workflow_service.components.tools_registry.is_feature_enabled")
    def test_generic_tools_not_available_with_feature_flag_disabled(
        self, mock_is_feature_enabled, tool_metadata
    ):
        """Test that generic tools are NOT available when feature flag is disabled."""
        mock_is_feature_enabled.return_value = False

        registry = ToolsRegistry(
            enabled_tools=["read_only_gitlab"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        # Verify generic tools are NOT present when feature flag is disabled
        assert "gitlab_api_get" not in registry._enabled_tools
        assert "gitlab_graphql" not in registry._enabled_tools

    def test_generic_tools_not_available_without_gitlab_privileges(self, tool_metadata):
        """Test that generic tools are NOT available without GitLab privileges."""
        registry = ToolsRegistry(
            enabled_tools=["read_write_files"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        # Verify generic tools are NOT present
        assert "gitlab_api_get" not in registry._enabled_tools
        assert "gitlab_graphql" not in registry._enabled_tools

    @patch("duo_workflow_service.components.tools_registry.is_feature_enabled")
    def test_generic_tools_follow_preapproval_rules(
        self, mock_is_feature_enabled, tool_metadata
    ):
        """Test that generic tools follow the same preapproval rules as other GitLab tools."""
        mock_is_feature_enabled.return_value = True

        # When generic tools privilege is preapproved, generic tools should be too
        registry_preapproved = ToolsRegistry(
            enabled_tools=["use_generic_gitlab_api_tools"],
            preapproved_tools=["use_generic_gitlab_api_tools"],
            tool_metadata=tool_metadata,
        )
        assert not registry_preapproved.approval_required("gitlab_api_get")
        assert not registry_preapproved.approval_required("gitlab_graphql")

        # When generic tools privilege is not preapproved, generic tools should require approval
        registry_not_preapproved = ToolsRegistry(
            enabled_tools=["use_generic_gitlab_api_tools"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )
        assert registry_not_preapproved.approval_required("gitlab_api_get")
        assert registry_not_preapproved.approval_required("gitlab_graphql")

    @patch("duo_workflow_service.components.tools_registry.is_feature_enabled")
    def test_generic_tools_can_be_used_in_toolset(
        self, mock_is_feature_enabled, tool_metadata
    ):
        """Test that generic tools can be requested in a toolset."""
        mock_is_feature_enabled.return_value = True

        registry = ToolsRegistry(
            enabled_tools=["read_only_gitlab"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        toolset = registry.toolset(["gitlab_api_get", "gitlab_graphql"])

        assert "gitlab_api_get" in toolset._all_tools
        assert "gitlab_graphql" in toolset._all_tools


class TestCapabilityDependentTools:
    """Tests for capability-dependent tools functionality."""

    @patch("duo_workflow_service.components.tools_registry.is_client_capable")
    def test_capability_dependent_tool_enabled_when_capable(
        self, mock_is_client_capable, tool_metadata
    ):
        """Test that capability-dependent tools are enabled when client has the capability."""
        mock_is_client_capable.return_value = True

        registry = ToolsRegistry(
            enabled_tools=["run_commands"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        # ShellCommand instead of RunCommand should be in enabled tools when capability is present
        assert "shell_command" in registry._enabled_tools
        assert "run_command" not in registry._enabled_tools

    @patch("duo_workflow_service.components.tools_registry.is_client_capable")
    def test_capability_dependent_tool_disabled_when_not_capable(
        self, mock_is_client_capable, tool_metadata
    ):
        """Test that capability-dependent tools are disabled when client lacks the capability."""
        mock_is_client_capable.return_value = False

        registry = ToolsRegistry(
            enabled_tools=["run_commands"],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        # RunCommand instead of ShellCommand should be in enabled tools when capability is present
        assert "shell_command" not in registry._enabled_tools
        assert "run_command" in registry._enabled_tools

    @patch("duo_workflow_service.components.tools_registry.is_client_capable")
    def test_capability_dependent_tool_disabled_when_enabled_but_missing_privilege(
        self, mock_is_client_capable, tool_metadata
    ):
        """Even if frontend sends capability, if agent privilege is missing, don't add it."""
        mock_is_client_capable.return_value = True

        registry = ToolsRegistry(
            enabled_tools=[],
            preapproved_tools=[],
            tool_metadata=tool_metadata,
        )

        # RunCommand instead of ShellCommand should be in enabled tools when capability is present
        assert "shell_command" not in registry._enabled_tools
        assert "run_command" not in registry._enabled_tools

    def test_capability_dependent_tool_missing_required_capability_raises_error(
        self, tool_metadata
    ):
        """Test that missing required_capability attribute raises RuntimeError."""
        mock_tool_cls = MagicMock(spec=BaseTool)
        mock_tool_cls.__name__ = "MockTool"
        (
            delattr(mock_tool_cls, "required_capability")
            if hasattr(mock_tool_cls, "required_capability")
            else None
        )

        with patch(
            "duo_workflow_service.components.tools_registry._CAPABILITY_DEPENDENT_TOOLS",
            [mock_tool_cls],
        ):
            with pytest.raises(
                RuntimeError,
                match="Tool MockTool is in _CAPABILITY_DEPENDENT_TOOLS but does not define 'required_capability'",
            ):
                ToolsRegistry(
                    enabled_tools=[],
                    preapproved_tools=[],
                    tool_metadata=tool_metadata,
                )

    @patch("duo_workflow_service.components.tools_registry.is_client_capable")
    def test_capability_dependent_tool_inherits_preapproval(
        self, mock_is_client_capable, tool_metadata
    ):
        """Test that capability-dependent tools are enabled when client has the capability."""
        mock_is_client_capable.return_value = True

        registry = ToolsRegistry(
            enabled_tools=["run_commands"],
            preapproved_tools=["run_commands"],
            tool_metadata=tool_metadata,
        )

        # ShellCommand instead of RunCommand should be in enabled tools when capability is present
        assert "shell_command" in registry._enabled_tools
        assert "shell_command" in registry._preapproved_tool_names
        assert "run_command" not in registry._enabled_tools
        assert "run_command" not in registry._preapproved_tool_names
