import hashlib
import json
import logging
from typing import Any, Optional, Sequence, Type, TypedDict, Union

from langchain.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service import tools
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.gitlab_api import Project, WorkflowConfig
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.tools import Toolset, ToolType
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
from duo_workflow_service.tools.mcp_tools import McpTool, McpToolConfig
from duo_workflow_service.tools.vulnerabilities.get_vulnerability_details import (
    EvaluateVulnerabilityFalsePositiveStatus,
    GetVulnerabilityDetails,
)
from duo_workflow_service.tools.vulnerabilities.post_sast_fp_analysis_to_gitlab import (
    PostSastFpAnalysisToGitlab,
)
from duo_workflow_service.tools.vulnerabilities.post_secret_fp_analysis_to_gitlab import (
    PostSecretFpAnalysisToGitlab,
)
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.language_server import LanguageServerVersion

logger = logging.getLogger(__name__)


class ToolMetadata(TypedDict):
    outbox: Outbox
    gitlab_client: GitlabHttpClient
    gitlab_host: str
    project: Optional[Project]


# This tools agent uses to interact with its internal state, they are required for
# a workflow to progress, and they do not pose any security risk, therefore they
# are being exempted from dynamic configuration.
_DEFAULT_TOOLS: list[Type[BaseTool]] = [
    tools.CreatePlan,
    tools.AddNewTask,
    tools.RemoveTask,
    tools.UpdateTaskDescription,
    tools.GetPlan,
    tools.SetTaskStatus,
]

# These tools are used to request formatted and definitive output from
# an agent. They can't be executed and they are not supposed to interact
# with any external systems, therefore they are being exempted from dynamic
# configuration.

NO_OP_TOOLS: list[Type[BaseModel]] = [
    tools.HandoverTool,
    tools.RequestUserClarificationTool,
]

# These tools require specific client capabilities to function properly.
# They are only enabled when the required capability is present.
_CAPABILITY_DEPENDENT_TOOLS: list[Type[BaseTool]] = [
    tools.ShellCommand,
]

_READ_ONLY_GITLAB_TOOLS: list[Type[BaseTool]] = [
    tools.ListIssues,
    tools.GetIssue,
    tools.GetLogsFromJob,
    tools.GetMergeRequest,
    tools.ListMergeRequest,
    tools.ListMergeRequestDiffs,
    tools.ListAllMergeRequestNotes,
    tools.GetPipelineFailingJobs,
    tools.GetDownstreamPipelines,
    tools.GetProject,
    tools.DocumentationSearch,
    tools.GroupProjectSearch,
    tools.IssueSearch,
    tools.MilestoneSearch,
    tools.UserSearch,
    tools.BlobSearch,
    tools.CommitSearch,
    tools.WikiBlobSearch,
    tools.NoteSearch,
    tools.GetEpic,
    tools.ListEpics,
    tools.ListIssueNotes,
    tools.GetIssueNote,
    tools.GetRepositoryFile,
    tools.ListRepositoryTree,
    tools.ListEpicNotes,
    tools.GetCommit,
    tools.ListCommits,
    tools.GetCommitDiff,
    tools.GetCommitComments,
    tools.GetSessionContext,
    tools.ListVulnerabilities,
    tools.CiLinter,
    tools.GetWorkItem,
    tools.ListWorkItems,
    tools.GetWorkItemNotes,
    tools.ListInstanceAuditEvents,
    tools.ListGroupAuditEvents,
    tools.ListProjectAuditEvents,
    tools.GetCurrentUser,
    GetVulnerabilityDetails,
    EvaluateVulnerabilityFalsePositiveStatus,
    tools.ExtractLinesFromText,
    tools.RunGLQLQuery,
    BuildReviewMergeRequestContext,
    GetSecurityFindingDetails,
    ListSecurityFindings,
    tools.GetWikiPage,
]

# Generic GitLab API tools - conditionally enabled via feature flag
_GENERIC_GITLAB_API_TOOLS: list[Type[BaseTool]] = [
    GitLabApiGet,
    GitLabGraphQL,
]

_RUN_MCP_TOOLS_PRIVILEGE = "run_mcp_tools"
_USE_GENERIC_GITLAB_API_TOOLS_PRIVILEGE = "use_generic_gitlab_api_tools"

# Using Sequence instead of list because it's covariant, allowing subclasses
ToolsOrConfigs = Union[Sequence[Type[BaseTool]], Sequence[McpToolConfig]]

_AGENT_PRIVILEGES: dict[str, list[Type[BaseTool]]] = {
    "read_write_files": [
        tools.ReadFile,
        tools.ReadFiles,
        tools.WriteFile,
        tools.EditFile,
        tools.ListDir,
        tools.FindFiles,
        tools.Grep,
        tools.Mkdir,
        tools.ExtractLinesFromText,
        tools.RunTests,
    ],
    "use_git": [
        tools.git.Command,
    ],
    "read_write_gitlab": [
        tools.UpdateVulnerabilitySeverity,
        tools.CreateIssue,
        tools.UpdateIssue,
        tools.CreateIssueNote,
        tools.CreateMergeRequest,
        tools.CreateMergeRequestNote,
        tools.UpdateMergeRequest,
        tools.CreateEpic,
        tools.UpdateEpic,
        tools.CreateCommit,
        tools.CreateBranch,
        tools.DismissVulnerability,
        tools.ConfirmVulnerability,
        tools.CreateWorkItem,
        tools.CreateWorkItemNote,
        tools.LinkVulnerabilityToIssue,
        tools.LinkVulnerabilityToMergeRequest,
        tools.UpdateWorkItem,
        tools.RevertToDetectedVulnerability,
        tools.CreateVulnerabilityIssue,
        PostSastFpAnalysisToGitlab,
        PostSecretFpAnalysisToGitlab,
        PostDuoCodeReview,
        *_READ_ONLY_GITLAB_TOOLS,
    ],
    "read_only_gitlab": _READ_ONLY_GITLAB_TOOLS,
    "run_commands": [
        tools.RunCommand,
    ],
    _RUN_MCP_TOOLS_PRIVILEGE: [],
    _USE_GENERIC_GITLAB_API_TOOLS_PRIVILEGE: _GENERIC_GITLAB_API_TOOLS,
}


class ToolsRegistry:
    _enabled_tools: dict[str, Union[BaseTool, Type[BaseModel]]]
    _preapproved_tool_names: set[str]
    _mcp_tool_names: list[str]
    _tool_call_approvals: dict

    @classmethod
    async def configure(
        cls,
        workflow_config: WorkflowConfig,
        gl_http_client: GitlabHttpClient,
        outbox: Outbox,
        project: Optional[Project],
        mcp_tools: Optional[list[McpToolConfig]] = None,
        language_server_version: Optional[LanguageServerVersion] = None,
    ):
        if not workflow_config:
            raise RuntimeError("Failed to find tools configuration for workflow")

        if "agent_privileges_names" not in workflow_config:
            raise RuntimeError(
                f"Failed to find tools configuration for workflow {workflow_config.get('id', 'None')}"
            )

        agent_privileges = workflow_config.get("agent_privileges_names", [])
        preapproved_tools = workflow_config.get(
            "pre_approved_agent_privileges_names", []
        )
        tool_metadata = ToolMetadata(
            outbox=outbox,
            gitlab_client=gl_http_client,
            gitlab_host=workflow_config.get("gitlab_host", ""),
            project=project,
        )

        return cls(
            enabled_tools=agent_privileges,
            preapproved_tools=preapproved_tools,
            tool_metadata=tool_metadata,
            mcp_tools=mcp_tools,
            language_server_version=language_server_version,
            tool_call_approvals=workflow_config.get("tool_call_approvals", {}),
        )

    def __init__(
        self,
        enabled_tools: list[str],
        preapproved_tools: list[str],
        tool_call_approvals: dict,
        tool_metadata: ToolMetadata,
        mcp_tools: Optional[list[McpToolConfig]] = None,
        language_server_version: Optional[LanguageServerVersion] = None,
    ):
        tools_for_agent_privileges: dict[str, ToolsOrConfigs] = dict(_AGENT_PRIVILEGES)

        # Always enable mcp tools until it's reliably passed by clients as an agent privilege
        enabled_tools.append(_RUN_MCP_TOOLS_PRIVILEGE)

        if _RUN_MCP_TOOLS_PRIVILEGE in enabled_tools:
            tools_for_agent_privileges[_RUN_MCP_TOOLS_PRIVILEGE] = mcp_tools or []

        # Conditionally enable generic GitLab API tools based on feature flag
        if (
            is_feature_enabled(FeatureFlag.USE_GENERIC_GITLAB_API_TOOLS)
            and _USE_GENERIC_GITLAB_API_TOOLS_PRIVILEGE not in enabled_tools
        ):
            enabled_tools.append(_USE_GENERIC_GITLAB_API_TOOLS_PRIVILEGE)

        self._enabled_tools = {
            **{tool_cls.tool_title: tool_cls for tool_cls in NO_OP_TOOLS},  # type: ignore
            **{tool.name: tool for tool in [tool_cls() for tool_cls in _DEFAULT_TOOLS]},
        }

        self._preapproved_tool_names = set(self._enabled_tools.keys())
        self._mcp_tool_names = [tool["llm_name"] for tool in mcp_tools or []]

        self._tool_call_approvals = tool_call_approvals

        for privilege in enabled_tools:
            for tool_cls_or_config in tools_for_agent_privileges.get(privilege, []):
                # Handle both regular tool classes and MCP tool configs
                if (
                    isinstance(tool_cls_or_config, dict)
                    and "llm_name" in tool_cls_or_config
                ):
                    tool = McpTool(
                        name=tool_cls_or_config["llm_name"],
                        description=tool_cls_or_config["description"],
                        args_schema=tool_cls_or_config["args_schema"],
                        metadata=tool_metadata,  # type: ignore[arg-type]
                    )
                    tool._original_mcp_name = tool_cls_or_config["original_name"]
                else:
                    tool = tool_cls_or_config(metadata=tool_metadata)  # type: ignore[assignment]

                # If language server client was detected, restrict tool versions
                if (
                    isinstance(tool, DuoBaseTool)
                    and language_server_version
                    and not language_server_version.supports_node_executor_tools()
                ):
                    continue

                self._enabled_tools[tool.name] = tool
                if privilege in preapproved_tools:
                    self._preapproved_tool_names.add(tool.name)

        # Add capability-dependent tools if condition is met
        for tool_cls in _CAPABILITY_DEPENDENT_TOOLS:
            # Access required_capability as a class variable (ClassVar)
            required_capability = getattr(tool_cls, "required_capability", None)
            if not required_capability:
                error_msg = (
                    f"Tool {tool_cls.__name__} is in "
                    "_CAPABILITY_DEPENDENT_TOOLS but does not define "
                    "'required_capability'"
                )
                raise RuntimeError(error_msg)

            # Don't add capability dependent tool if it supersedes another tool
            # and agent privilege for the superseded tool is missing
            supersedes = getattr(tool_cls, "supersedes", None)
            if (
                supersedes
                and supersedes.model_fields["name"].default not in self._enabled_tools
            ):
                continue

            if is_client_capable(required_capability):
                # replace the superseded tool to supersedes' tool instance
                # pre-approval status will not be changed
                self._enabled_tools[tool_cls.model_fields["name"].default] = tool_cls(
                    metadata=tool_metadata
                )

    def get(self, tool_name: str) -> Optional[ToolType]:
        return self._enabled_tools.get(tool_name)

    def get_batch(self, tool_names: list[str]) -> list[ToolType]:
        return [
            self._enabled_tools[tool_name]
            for tool_name in tool_names
            if tool_name in self._enabled_tools
        ]

    def get_handlers(self, tool_names: list[str]) -> list[BaseTool]:
        tool_handlers: list[BaseTool] = []
        for tool_name in tool_names:
            handler = self._enabled_tools.get(tool_name)
            if isinstance(handler, BaseTool):
                tool_handlers.append(handler)

        return tool_handlers

    def approval_required(
        self, tool_name: str, tool_args: dict[Any, Any] | None = None
    ) -> bool:
        """Check if a tool requires human approval before execution.

        Args:
            tool_name: The name of the tool to check
            tool_args: The arguments passed to the tool

        Returns:
            False if the tool is in the preapproved list,
            True otherwise.
        """
        if tool_args is None:
            tool_args = {}

        if tool_name in self._preapproved_tool_names:
            return False

        approved_tool_calls_config = self._tool_call_approvals.get(tool_name, None)
        if approved_tool_calls_config is None:
            return True

        if "call_args" not in approved_tool_calls_config:
            logger.warning(
                "Approved tool calls do not comply with expected format",
                extra={
                    "approved_tool_calls": approved_tool_calls_config,
                },
            )
            return True

        approved_tool_calls = approved_tool_calls_config["call_args"]

        # Convert tool_args to SHA256 hexdigest for comparison
        # Match Rails JSON serialization: compact format
        tool_args_json = json.dumps(tool_args, separators=(",", ":"))
        tool_args_hash = hashlib.sha256(tool_args_json.encode()).hexdigest()

        logger.debug(
            "Searching for tool call approval",
            extra={
                "tool_name": tool_name,
                "tool_args_hash": tool_args_hash,
                "tool_call_approvals": approved_tool_calls,
            },
        )

        return not any(
            tool_args_hash == approved_call_hash
            for approved_call_hash in approved_tool_calls
        )

    def toolset(self, tool_names: list[str]) -> Toolset:
        """Create a Toolset instance representing complete collection of tools available to an agent.

        Args:
            tool_names: A list of tool names to include in the Toolset.

        Returns:
            A new Toolset instance containing the requested tools.
        """

        # MCP tools if there are any are added to toolset
        tool_names += self._mcp_tool_names

        all_tools = {
            tool_name: self._enabled_tools[tool_name]
            for tool_name in tool_names
            if tool_name in self._enabled_tools
        }

        pre_approved = {
            tool_name
            for tool_name in tool_names
            if tool_name in self._preapproved_tool_names
        }

        return Toolset(pre_approved=pre_approved, all_tools=all_tools)
