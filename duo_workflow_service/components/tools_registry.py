import copy
import json
from typing import Any, NotRequired, Optional, Sequence, Type, TypedDict, Union

import structlog
from langchain.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service import tools
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.gitlab_api import (
    Project,
    WorkflowConfig,
    WorkflowFeatures,
    workflow_global_id,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.tools import Toolset, ToolType
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_api_generic import GitLabApiGet, GitLabGraphQL
from duo_workflow_service.tools.mcp_tools import McpTool, McpToolConfig
from duo_workflow_service.tools.mr_discussions import (
    ListMrDiscussions,
    ReplyToDiscussion,
    SetDiscussionResolved,
)
from duo_workflow_service.tools.mr_review import SubmitMrReview
from duo_workflow_service.tools.set_form_permissions import SetFormPermissions
from duo_workflow_service.tools.update_form_fields import UpdateFormFields
from duo_workflow_service.tools.update_form_permissions import UpdateFormPermissions
from lib.language_server import LanguageServerVersion

log = structlog.stdlib.get_logger("tools_registry")


class ToolMetadata(TypedDict):
    workflow_id: str
    outbox: Outbox
    gitlab_client: GitlabHttpClient
    gitlab_host: str
    project: Optional[Project]
    features: NotRequired[WorkflowFeatures]


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
    tools.TodoWrite,
    tools.ClarificationQuestionTool,
    tools.RenderUiTool,
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
# They are only enabled when all capabilities in their required_capability frozenset are present.
_CAPABILITY_DEPENDENT_TOOLS: list[Type[BaseTool]] = [
    tools.ShellCommand,
    tools.AdvanceBlobSearch,
    tools.ReadFileChunked,
]

_READ_ONLY_FILE_TOOLS: list[Type[BaseTool]] = [
    tools.ReadFile,
    tools.ReadFiles,
    tools.ListDir,
    tools.FindFiles,
    tools.Grep,
    tools.ExtractLinesFromText,
]

_READ_ONLY_GITLAB_TOOLS: list[Type[BaseTool]] = [
    tools.ListIssues,
    tools.GetIssue,
    tools.GetLogsFromJob,
    tools.GetMergeRequest,
    tools.ListMergeRequest,
    tools.ListMergeRequestDiffs,
    tools.ListAllMergeRequestNotes,
    ListMrDiscussions,
    tools.GetPipelineFailingJobs,
    tools.GetDownstreamPipelines,
    tools.GetFailingBridgeJobs,
    tools.GetProject,
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
    tools.GetWorkItemStatuses,
    tools.ListInstanceAuditEvents,
    tools.ListGroupAuditEvents,
    tools.ListProjectAuditEvents,
    tools.GetCurrentUser,
    tools.GetVulnerabilityDetails,
    tools.EvaluateVulnerabilityFalsePositiveStatus,
    tools.ExtractLinesFromText,
    tools.GetGlqlSchema,
    tools.RunGLQLQuery,
    tools.BuildReviewMergeRequestContext,
    tools.GetSecurityFindingDetails,
    tools.ListSecurityFindings,
    tools.ListAscpScans,
    tools.ListAscpComponents,
    tools.GetWikiPage,
    tools.DocumentationSearch,
    GitLabApiGet,
    GitLabGraphQL,
]

_RUN_MCP_TOOLS_PRIVILEGE = "run_mcp_tools"

# Using Sequence instead of list because it's covariant, allowing subclasses
ToolsOrConfigs = Union[Sequence[Type[BaseTool]], Sequence[McpToolConfig]]

_AGENT_PRIVILEGES: dict[str, list[Type[BaseTool]]] = {
    "read_only_files": _READ_ONLY_FILE_TOOLS,
    "read_write_files": [
        *_READ_ONLY_FILE_TOOLS,
        tools.WriteFile,
        tools.EditFile,
        tools.Mkdir,
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
        tools.CreateMergeRequestDiffNote,
        tools.UpdateMergeRequest,
        tools.CreateEpic,
        tools.UpdateEpic,
        tools.CreateCommit,
        tools.CreateBranch,
        tools.DismissVulnerability,
        tools.ConfirmVulnerability,
        tools.CreateWorkItem,
        tools.CreateWorkItemNote,
        tools.CreateAscpScan,
        tools.CreateAscpComponent,
        tools.CreateAscpSecurityContext,
        tools.LinkVulnerabilityToIssue,
        tools.LinkVulnerabilityToMergeRequest,
        tools.UpdateWorkItem,
        tools.RevertToDetectedVulnerability,
        tools.CreateVulnerabilityIssue,
        tools.PostSastFpAnalysisToGitlab,
        tools.PostSecretFpAnalysisToGitlab,
        tools.PostDuoCodeReview,
        SubmitMrReview,
        ReplyToDiscussion,
        SetDiscussionResolved,
        UpdateFormFields,
        UpdateFormPermissions,
        SetFormPermissions,
        *_READ_ONLY_GITLAB_TOOLS,
    ],
    "read_only_gitlab": _READ_ONLY_GITLAB_TOOLS,
    "run_commands": [
        tools.RunCommand,
    ],
    _RUN_MCP_TOOLS_PRIVILEGE: [],
    "start_flows": [
        tools.StartFlow,
    ],
}


TOOL_CALL_APPROVED_QUERY = """
query($workflowId: AiDuoWorkflowsWorkflowID!, $toolName: String!, $toolCallArgs: String!) {
    duoWorkflowWorkflows(workflowId: $workflowId) {
        nodes {
            toolCallApproved(toolName: $toolName, toolCallArgs: $toolCallArgs)
        }
    }
}
"""


class ToolsRegistry:
    _enabled_tools: dict[str, Union[BaseTool, Type[BaseModel]]]
    _preapproved_tool_names: set[str]
    _mcp_tool_names: list[str]

    @classmethod
    async def configure(
        cls,
        workflow_config: WorkflowConfig,
        gl_http_client: GitlabHttpClient,
        outbox: Outbox,
        project: Optional[Project],
        workflow_id: Optional[str] = None,
        mcp_tools: Optional[list[McpToolConfig]] = None,
        language_server_version: Optional[LanguageServerVersion] = None,
        denied_tools: Optional[list[str]] = None,
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
            workflow_id=workflow_config.get("workflow_id", ""),
            outbox=outbox,
            gitlab_client=gl_http_client,
            gitlab_host=workflow_config.get("gitlab_host", ""),
            project=project,
            features=workflow_config.get("features", {}),
        )

        return cls(
            enabled_tools=agent_privileges,
            preapproved_tools=preapproved_tools,
            tool_metadata=tool_metadata,
            mcp_tools=mcp_tools,
            language_server_version=language_server_version,
            denied_tools=denied_tools or [],
            gl_http_client=gl_http_client,
            workflow_id=workflow_id,
        )

    def __init__(
        self,
        enabled_tools: list[str],
        preapproved_tools: list[str],
        tool_metadata: ToolMetadata,
        gl_http_client: Optional[GitlabHttpClient] = None,
        workflow_id: Optional[str] = None,
        mcp_tools: Optional[list[McpToolConfig]] = None,
        language_server_version: Optional[LanguageServerVersion] = None,
        denied_tools: Optional[list[str]] = None,
    ):
        tools_for_agent_privileges: dict[str, ToolsOrConfigs] = dict(_AGENT_PRIVILEGES)

        # Always enable mcp tools until it's reliably passed by clients as an agent privilege
        enabled_tools.append(_RUN_MCP_TOOLS_PRIVILEGE)

        if _RUN_MCP_TOOLS_PRIVILEGE in enabled_tools:
            tools_for_agent_privileges[_RUN_MCP_TOOLS_PRIVILEGE] = mcp_tools or []

        self._enabled_tools = {
            **{tool_cls.tool_title: tool_cls for tool_cls in NO_OP_TOOLS},  # type: ignore
            **{tool.name: tool for tool in [tool_cls() for tool_cls in _DEFAULT_TOOLS]},
        }

        self._preapproved_tool_names = set(self._enabled_tools.keys())
        self._denied_tools: set[str] = set(denied_tools or [])
        self._mcp_tool_names = [tool["llm_name"] for tool in mcp_tools or []]

        self._gl_http_client = gl_http_client
        self._workflow_id = workflow_id
        self._approved_cache: set[tuple[str, str]] = set()

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

        # Add capability-dependent tools if condition is met.
        # Each tool's required_capability frozenset encodes all capabilities needed,
        # so is_client_capable alone resolves to the correct supersession tool.
        for tool_cls in _CAPABILITY_DEPENDENT_TOOLS:
            required_capability = getattr(tool_cls, "required_capability", None)
            if not required_capability:
                error_msg = (
                    f"Tool {tool_cls.__name__} is in "
                    "_CAPABILITY_DEPENDENT_TOOLS but does not define "
                    "'required_capability'"
                )
                raise RuntimeError(error_msg)

            # Skip tool if client capability check fails
            if not is_client_capable(required_capability):
                continue

            # Don't add capability dependent tool if it supersedes another tool
            # and the superseded tool is not currently enabled
            supersedes = getattr(tool_cls, "supersedes", None)
            if (
                supersedes is not None
                and supersedes.model_fields["name"].default not in self._enabled_tools
            ):
                continue

            # replace the superseded tool with this tool instance
            # pre-approval status will not be changed
            # deny status is inherited implicitly: if the superseded tool name is in
            # _denied_tools, toolset() will strip it regardless of which instance is active
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

    def is_preapproved(self, tool_name: str) -> bool:
        """Check if a tool is preapproved (no approval ever needed).

        This is a local check against the privilege-level preapproval list. Use this for routing decisions where you
        don't need per-call checks.
        """
        return tool_name in self._preapproved_tool_names

    async def approval_required(
        self, tool_name: str, tool_args: dict[Any, Any] | None = None
    ) -> bool:
        """Check if a specific tool call requires human approval.

        Preapproved tools are checked locally. For all other tools,
        delegates to Rails which owns the approval logic (hash matching,
        pattern matching, match-target extraction).

        Args:
            tool_name: The name of the tool to check
            tool_args: The arguments passed to the tool

        Returns:
            False if the tool is approved, True if approval is required.
        """
        if tool_args is None:
            tool_args = {}

        if tool_name in self._preapproved_tool_names:
            return False

        if not self._gl_http_client or not self._workflow_id:
            return True

        try:
            tool_args_json = json.dumps(
                tool_args, sort_keys=True, separators=(",", ":")
            )
        except TypeError:
            log.warning(
                "Tool args not JSON-serializable, defaulting to requiring approval",
                extra={
                    "tool_name": tool_name,
                    "arg_types": {k: type(v).__name__ for k, v in tool_args.items()},
                },
            )
            return True

        cache_key = (tool_name, tool_args_json)
        if cache_key in self._approved_cache:
            log.debug(
                "Tool call approved from cache",
                extra={"tool_name": tool_name},
            )
            return False

        try:
            response = await self._gl_http_client.graphql(
                TOOL_CALL_APPROVED_QUERY,
                {
                    "workflowId": workflow_global_id(self._workflow_id),
                    "toolName": tool_name,
                    "toolCallArgs": tool_args_json,
                },
            )

            nodes = response.get("duoWorkflowWorkflows", {}).get("nodes", [])
            if nodes:
                approved = nodes[0].get("toolCallApproved", False)
                if approved:
                    self._approved_cache.add(cache_key)
                return not approved

            log.warning(
                "GraphQL returned empty nodes for tool approval check",
                extra={"tool_name": tool_name, "workflow_id": self._workflow_id},
            )

        # graphql() raises bare Exceptions for timeout/decode/query errors
        except Exception as exc:
            log.warning(
                "Failed to check tool approval via GraphQL, defaulting to requiring approval",
                extra={"tool_name": tool_name, "workflow_id": self._workflow_id},
                exc_info=exc,
            )

        return True

    def toolset(
        self,
        tool_names: list[str],
        tool_options: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Toolset:
        """Create a Toolset instance representing complete collection of tools available to an agent.

        Args:
            tool_names: A list of tool names to include in the Toolset.
            tool_options: Optional dict mapping tool names to their option overrides.
                Example: {"create_merge_request_note": {"force_internal": True}}

        Returns:
            A new Toolset instance containing the requested tools.
        """

        # MCP tools if there are any are added to toolset
        tool_names += self._mcp_tool_names

        if self._denied_tools:
            tool_names = [t for t in tool_names if t not in self._denied_tools]

        all_tools = {}
        for tool_name in tool_names:
            if tool_name not in self._enabled_tools:
                continue
            tool = self._enabled_tools[tool_name]
            # Clone tools that have options to avoid shared-state mutation
            # across different Toolsets using the same tool with different options
            if (
                tool_options
                and tool_name in tool_options
                and isinstance(tool, BaseTool)
            ):
                tool = copy.copy(tool)
                # Prevent metadata cross-contamination if mutated
                if hasattr(tool, "metadata") and tool.metadata:
                    tool.metadata = dict(tool.metadata)
            all_tools[tool_name] = tool

        pre_approved = {
            tool_name
            for tool_name in tool_names
            if tool_name in self._preapproved_tool_names
        }

        return Toolset(
            pre_approved=pre_approved,
            all_tools=all_tools,
            tool_options=tool_options or {},
        )
