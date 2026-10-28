import copy
import importlib.util
import json
import keyword
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Literal,
    NamedTuple,
    NotRequired,
    Optional,
    Sequence,
    Type,
    TypedDict,
    Union,
)

import boto3
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
from duo_workflow_service.tools.web_search import web_search_boto_session
from lib.feature_roots import default_features_dir
from lib.language_server import LanguageServerVersion

log = structlog.stdlib.get_logger("tools_registry")


class ToolMetadata(TypedDict):
    workflow_id: str
    outbox: Outbox
    gitlab_client: GitlabHttpClient
    gitlab_host: str
    project: Optional[Project]
    features: NotRequired[WorkflowFeatures]
    # Signs AgentCore gateway requests for web search. None whenever web search is disabled,
    # which AgentCoreWebSearch reports as a configuration error rather than searching unsigned.
    boto_session: NotRequired[Optional[boto3.Session]]


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
    tools.GrepLiteral,
    tools.NotifyMeWhen,
    tools.AgentCoreWebSearch,
]

# Capability-dependent tools that run without asking the user each time.
#
# Opt-in, and deliberately not derived from any other property of the tool: pre-approval is a
# security decision, so a tool added to `_CAPABILITY_DEPENDENT_TOOLS` must require approval until
# someone puts it here on purpose. Superseding tools are absent because they inherit the approval
# state of the tool they replace.
#
# Only tools with no effect outside this session belong here: a read, or a message the
# agent sends its future self.
_PREAPPROVED_CAPABILITY_TOOLS: list[Type[BaseTool]] = [
    # Reads public web pages and returns text. Nothing is written, and the agent is told to treat
    # results as untrusted, so prompting on each search would only add friction.
    tools.AgentCoreWebSearch,
    # A message from the agent to its future self. Touches no file, process, network or
    # GitLab resource, so prompting on it is friction without protection. See #2906.
    tools.NotifyMeWhen,
]

_READ_ONLY_FILE_TOOLS: list[Type[BaseTool]] = [
    tools.ReadFile,
    tools.ReadFiles,
    tools.ListDir,
    tools.FindFiles,
    tools.Grep,
    tools.ExtractLinesFromText,
    tools.FinalizeCodeReviewFindings,
    tools.DetectReviewContextTruncation,
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
    tools.GetRepositoryFiles,
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
    tools.ResolveVulnerabilityTargetBranch,
    tools.ExtractLinesFromText,
    tools.FetchGlqlSchema,
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

# Why ``ToolsRegistry.toolset`` dropped a requested tool; see ``_unavailable_reason``.
UnavailableReason = Literal[
    "flow_restricted", "missing_privilege", "unsupported_client", "unknown"
]

# Using Sequence instead of list because it's covariant, allowing subclasses
ToolsOrConfigs = Union[Sequence[Type[BaseTool]], Sequence[McpToolConfig]]

_AGENT_PRIVILEGES: dict[str, list[Type[BaseTool]]] = {
    "read_only_files": _READ_ONLY_FILE_TOOLS,
    "read_write_files": [
        tools.BlPrioritizeAndCap,
        *_READ_ONLY_FILE_TOOLS,
        tools.BlCollectAndFlatten,
        tools.WriteFile,
        tools.EditFile,
        tools.BlDedupFindings,
        tools.BlWriteSastReport,
        tools.Mkdir,
        tools.RunTests,
        tools.BlDiscoverAndCluster,
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
        tools.AddMergeRequestReviewers,
        tools.PostSuggestedReviewers,
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
        tools.PostDuoCodeReviewFindings,
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


def discover_feature_tools(
    features_dir: Optional[Path] = None,
) -> dict[str, list[Type[BaseTool]]]:
    """Collect ``{privilege: [tool classes]}`` declared by features under ai/features/.

    A feature that owns tools exposes a ``FEATURE_TOOLS`` mapping in its
    ``components`` package. Silent when the tree is absent.

    Args:
        features_dir: Root directory to scan instead of the real
            ``ai/features`` tree. Intended for tests; defaults to the actual
            features directory when omitted.

    Returns:
        A mapping of privilege name to the list of tool classes declared
        for it.
    """
    root = features_dir or default_features_dir()
    discovered: dict[str, list[Type[BaseTool]]] = {}
    if not root.is_dir():
        return discovered

    for init_file in sorted(root.glob("*/*/components/__init__.py")):
        feature = init_file.parent.parent.name
        domain = init_file.parent.parent.parent.name
        # A non-identifier or keyword name (e.g. "evil.pkg", "class") cannot
        # form an importable package name.
        if not (
            domain.isidentifier()
            and feature.isidentifier()
            and not keyword.iskeyword(domain)
            and not keyword.iskeyword(feature)
        ):
            log.warning(
                "Skipping feature with non-identifier directory name",
                domain=domain,
                feature=feature,
            )
            continue
        if features_dir is None:
            # Real tree: the normal import machinery binds parent packages and
            # caches in sys.modules, keeping tool classes identity-stable for
            # any later plain import of the components package.
            module = importlib.import_module(
                f"ai.features.{domain}.{feature}.components"
            )
        else:
            # Synthetic tree (tests): execute a throwaway module from the file
            # without touching sys.modules or the real package namespace.
            spec = importlib.util.spec_from_file_location(
                f"_feature_tools_{domain}_{feature}", init_file
            )
            if spec is None or spec.loader is None:
                log.warning(
                    "Skipping feature whose components package has no loadable spec",
                    domain=domain,
                    feature=feature,
                )
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        for privilege, tool_classes in getattr(module, "FEATURE_TOOLS", {}).items():
            discovered.setdefault(privilege, []).extend(tool_classes)
    return discovered


# read_write_gitlab is a superset of read_only_gitlab, so a read-only tool joins both.
_PRIVILEGE_SUPERSETS: dict[str, list[str]] = {
    "read_only_gitlab": ["read_only_gitlab", "read_write_gitlab"],
    "read_only_files": ["read_only_files", "read_write_files"],
}


def _tool_name(tool_cls: Union[Type[BaseTool], Type[BaseModel]]) -> str:
    """Registry name of a tool class; no-op tools use ``tool_title`` instead of a ``name`` field."""
    title = getattr(tool_cls, "tool_title", None)
    if title is not None:
        return title
    name = tool_cls.model_fields["name"].default
    if not isinstance(name, str):
        raise ValueError(
            f"{tool_cls.__name__} declares no default for its name field, so it "
            "cannot be checked for registry collisions"
        )
    return name


def allowed_flows_for_tool(tool_name: str) -> Optional[frozenset[str]]:
    """The ``allowed_flows`` of the privilege-granted tool named ``tool_name``; ``None`` when any flow may use it."""
    for tool_classes in _AGENT_PRIVILEGES.values():
        for tool_cls in tool_classes:
            if _tool_name(tool_cls) == tool_name:
                return getattr(tool_cls, "allowed_flows", None)
    return None


def agent_privileges_for_tool(tool_name: str) -> list[str]:
    """Names of the agent privileges that grant ``tool_name``; empty when no privilege does (e.g. MCP tools)."""
    return [
        privilege
        for privilege, tool_classes in _AGENT_PRIVILEGES.items()
        if any(_tool_name(tool_cls) == tool_name for tool_cls in tool_classes)
    ]


class _ToolGrant(NamedTuple):
    privileges: set[str]
    allowed_flows: Optional[frozenset[str]]


def _index_tool_grants() -> dict[str, _ToolGrant]:
    """Map each privilege-granted tool name to its granting privileges and ``allowed_flows``, in one pass."""
    grants: dict[str, _ToolGrant] = {}
    for privilege, tool_classes in _AGENT_PRIVILEGES.items():
        for tool_cls in tool_classes:
            name = _tool_name(tool_cls)
            if name not in grants:
                grants[name] = _ToolGrant(
                    set(), getattr(tool_cls, "allowed_flows", None)
                )
            grants[name].privileges.add(privilege)
    return grants


def _merge_feature_tools(
    privileges: dict[str, list[Type[BaseTool]]],
    discovered: dict[str, list[Type[BaseTool]]],
    reserved: Iterable[Union[Type[BaseTool], Type[BaseModel]]] = (),
) -> None:
    """Merge feature-declared tools into the privilege map.

    A feature tool must declare a privilege that already exists in the map and a tool name no other tool uses, so a typo
    cannot leave a tool silently ungated or shadow another tool in the registry.
    """
    taken = {_tool_name(tc) for tcs in privileges.values() for tc in tcs}
    taken.update(_tool_name(tc) for tc in reserved)

    for privilege, tool_classes in discovered.items():
        if privilege == _RUN_MCP_TOOLS_PRIVILEGE:
            raise ValueError(
                f"Feature tools {[_tool_name(tc) for tc in tool_classes]} declare "
                f"{privilege!r}, which the registry rebuilds at runtime; tools "
                "declared under it would be silently dropped"
            )
        if privilege not in privileges:
            raise ValueError(
                f"Feature tools {[_tool_name(tc) for tc in tool_classes]} declare "
                f"unknown privilege {privilege!r}; known privileges: "
                f"{sorted(privileges)}"
            )
        for tool_cls in tool_classes:
            name = _tool_name(tool_cls)
            if name in taken:
                raise ValueError(
                    f"Feature tool name {name!r} is already registered; "
                    "tool names must be unique across the registry"
                )
            taken.add(name)
        for target in _PRIVILEGE_SUPERSETS.get(privilege, [privilege]):
            # Rebind instead of extending: the privilege lists alias module-level
            # constants, which an in-place extend would silently mutate.
            # The taken-name guard above already rejected duplicates.
            privileges[target] = privileges.get(target, []) + list(tool_classes)


def _reserved_tool_names(
    privileges: dict[str, ToolsOrConfigs],
) -> set[str]:
    """Names no MCP tool may take, whatever origin it claims."""
    candidates: list[Union[Type[BaseTool], Type[BaseModel]]] = [
        *_DEFAULT_TOOLS,
        *NO_OP_TOOLS,
        *_CAPABILITY_DEPENDENT_TOOLS,
    ]
    for privilege, entries in privileges.items():
        if privilege == _RUN_MCP_TOOLS_PRIVILEGE:
            continue
        candidates.extend(e for e in entries if not isinstance(e, dict))

    reserved = set()
    for tool_cls in candidates:
        try:
            reserved.add(_tool_name(tool_cls))
        except ValueError:
            continue
    return reserved


def _privilege_tool_names(*privileges: str) -> set[str]:
    """Static tool names in these privilege groups, read at call time so merged feature tools count."""
    names = set()
    for privilege in privileges:
        for entry in _AGENT_PRIVILEGES.get(privilege, []):
            if isinstance(entry, dict):
                continue
            try:
                names.add(_tool_name(entry))
            except ValueError:
                continue
    return names


def ceiling_exempt_tool_names(
    ruled: Iterable[str], *, keep_read_only: bool = True
) -> set[str]:
    """Tools the governance ceiling keeps pre-approved although Rails cannot put them in allow.

    `ruled` is the claim's raw ask and deny lists. Rails always pre-approves start_flows, and its read-only groups
    default to allow on every surface, so DWS tools Rails cannot name there would otherwise prompt. A rule on any
    non-write tool may have demoted a read-only group in Rails, so it turns the read-only part off: at worst an
    extra prompt, never an extra pre-approval. Pass `keep_read_only=False` when the claim's allow list is empty:
    had a read-only group resolved to allow in Rails, its tools would be there.
    """
    ruled = set(ruled)
    read_only_names = _privilege_tool_names("read_only_gitlab", "read_only_files")
    # The write groups also hold the read tools, so only the names beyond those count as write.
    write_only = (
        _privilege_tool_names(
            "read_write_gitlab", "read_write_files", "run_commands", "use_git"
        )
        - read_only_names
    )
    exempt = _privilege_tool_names("start_flows") - ruled
    if keep_read_only and not ruled - write_only:
        exempt |= read_only_names - ruled
    return exempt


def _drop_shadowing_mcp_tools(
    mcp_tools: list[McpToolConfig], reserved: set[str], governed_names: set[str]
) -> list[McpToolConfig]:
    """Drop MCP tools that would answer for a tool GitLab owns.

    A reserved name is refused whatever the entry claims about its origin, because
    client_injected is only trustworthy behind Workhorse. Tools from a server GitLab
    configured are always server-prefixed, so they never hold a reserved name anyway.

    A governed name is refused only for client-injected entries: an admin rule naming a
    GitLab MCP tool must make it prompt, not remove it.

    On a name collision the non-client-injected entry wins. Workhorse already orders the
    lists so that it does, but that contract lives in another repo, so enforce it here
    rather than depend on it.
    """
    kept: dict[str, McpToolConfig] = {}
    for config in mcp_tools:
        incumbent = kept.get(config["llm_name"])
        if config["llm_name"] in reserved:
            refusal_reason = "reserved"
        elif config["client_injected"] and config["llm_name"] in governed_names:
            refusal_reason = "governed"
        elif (
            incumbent is not None
            and config["client_injected"]
            and not incumbent["client_injected"]
        ):
            refusal_reason = "collision"
        else:
            refusal_reason = None

        if refusal_reason:
            log.warning(
                "Dropped MCP tool shadowing a name GitLab owns",
                extra={
                    "original_name": config["original_name"],
                    "llm_name": config["llm_name"],
                    "client_injected": config["client_injected"],
                    "refusal_reason": refusal_reason,
                },
            )
            continue
        kept[config["llm_name"]] = config
    return list(kept.values())


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
    _ask_tool_names: set[str]
    _policy_permitted_preapprovals: Optional[set[str]]
    _mcp_tool_names: list[str]
    _enabled_privileges: set[str]

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
        ask_tools: Optional[list[str]] = None,
        allow_client_injected_mcp_tools: bool = False,
        allowed_tools: Optional[list[str]] = None,
        admin_allowed_preapprovals: Optional[set[str]] = None,
        registry_flow_id: Optional[str] = None,
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
            boto_session=web_search_boto_session(),
        )

        return cls(
            enabled_tools=agent_privileges,
            preapproved_tools=preapproved_tools,
            tool_metadata=tool_metadata,
            mcp_tools=mcp_tools,
            language_server_version=language_server_version,
            denied_tools=denied_tools or [],
            ask_tools=ask_tools or [],
            allow_client_injected_mcp_tools=allow_client_injected_mcp_tools,
            allowed_tools=allowed_tools or [],
            admin_allowed_preapprovals=admin_allowed_preapprovals,
            gl_http_client=gl_http_client,
            workflow_id=workflow_id,
            registry_flow_id=registry_flow_id,
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
        ask_tools: Optional[list[str]] = None,
        allow_client_injected_mcp_tools: bool = False,
        allowed_tools: Optional[list[str]] = None,
        admin_allowed_preapprovals: Optional[set[str]] = None,
        registry_flow_id: Optional[str] = None,
    ):
        """Build the tools a workflow can use.

        ``registry_flow_id`` is the Flow Registry id of the config the workflow runs, ``None`` for any other config.
        """
        tools_for_agent_privileges: dict[str, ToolsOrConfigs] = dict(_AGENT_PRIVILEGES)

        # Always enable mcp tools until it's reliably passed by clients as an agent privilege
        enabled_tools.append(_RUN_MCP_TOOLS_PRIVILEGE)
        self._enabled_privileges = set(enabled_tools)

        registered_mcp_tools: list[McpToolConfig] = []
        if _RUN_MCP_TOOLS_PRIVILEGE in enabled_tools:
            registered_mcp_tools = _drop_shadowing_mcp_tools(
                mcp_tools or [],
                _reserved_tool_names(tools_for_agent_privileges),
                set(ask_tools or []) | set(denied_tools or []),
            )
            tools_for_agent_privileges[_RUN_MCP_TOOLS_PRIVILEGE] = registered_mcp_tools

        self._enabled_tools = {
            **{tool_cls.tool_title: tool_cls for tool_cls in NO_OP_TOOLS},  # type: ignore
            **{tool.name: tool for tool in [tool_cls() for tool_cls in _DEFAULT_TOOLS]},
        }

        # Not privilege-derived, so the governance ceiling never strips them.
        baseline_preapproved = set(self._enabled_tools.keys())
        self._preapproved_tool_names = set(baseline_preapproved)
        self._denied_tools: set[str] = set(denied_tools or [])
        self._ask_tool_names: set[str] = set(ask_tools or [])
        self._mcp_tool_names = [tool["llm_name"] for tool in registered_mcp_tools]

        self._gl_http_client = gl_http_client
        self._workflow_id = workflow_id
        self._approved_cache: set[tuple[str, str]] = set()

        self._registry_flow_id = registry_flow_id
        for privilege in enabled_tools:
            for tool_cls_or_config in tools_for_agent_privileges.get(privilege, []):
                allowed_flows = getattr(tool_cls_or_config, "allowed_flows", None)
                if allowed_flows is not None and registry_flow_id not in allowed_flows:
                    continue
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
                    tool._client_injected = tool_cls_or_config["client_injected"]
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

            # Skip tool if it declares additional runtime conditions that are not met,
            # such as a feature flag or a fallback that only applies for some models.
            is_available = getattr(tool_cls, "is_available", None)
            if is_available is not None and not is_available():
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
            tool_name = tool_cls.model_fields["name"].default
            self._enabled_tools[tool_name] = tool_cls(metadata=tool_metadata)

            # A tool that supersedes nothing introduces a new name, so there is no pre-approval to
            # inherit and the default is to prompt. Pre-approval is granted only by explicit
            # listing, so adding a capability-dependent tool never changes its approval behaviour
            # by itself.
            if tool_cls in _PREAPPROVED_CAPABILITY_TOOLS:
                self._preapproved_tool_names.add(tool_name)
                baseline_preapproved.add(tool_name)

        if allow_client_injected_mcp_tools:
            client_injected = {
                name
                for name, tool in self._enabled_tools.items()
                if getattr(tool, "_client_injected", False)
            }
            self._preapproved_tool_names.update(client_injected)
            # A client names its own MCP tools, so no admin allow list can rule on them.
            baseline_preapproved.update(client_injected)

        # Per-tool admin `allow` rules pre-approve a tool even when its privilege group
        # is not pre-approved as a whole (a mixed group). Scoped to enabled tools so an
        # allow cannot introduce a tool the flow never had, and to GitLab-origin tools:
        # a client names its own MCP tools, so an allow on a GitLab MCP tool name must
        # not reach a client entry that shares it. The flag above covers the opted-in case.
        self._preapproved_tool_names |= {
            name
            for name in allowed_tools or []
            if name in self._enabled_tools
            and not getattr(self._enabled_tools[name], "_client_injected", False)
        }

        # The workflow row is writable by the session owner, so the pre-approvals derived
        # from it are capped at what admin policy allows. `None` means the ceiling is off.
        # An allowed name only admits a GitLab-origin tool, matching the union above.
        if admin_allowed_preapprovals is None:
            self._policy_permitted_preapprovals = None
        else:
            self._policy_permitted_preapprovals = {
                name
                for name in admin_allowed_preapprovals
                if not getattr(self._enabled_tools.get(name), "_client_injected", False)
            } | baseline_preapproved
        removed = self._preapproved_tool_names - self.pre_approvals_allowed_by_policy(
            self._preapproved_tool_names
        )
        if removed:
            log.info(
                "Removed tools from pre-approval: not permitted by admin policy",
                workflow_id=workflow_id,
                tool_names=sorted(removed),
            )
        self._preapproved_tool_names -= removed

        # Applied last so it outranks every pre-approval source above. Only binds where
        # a user can be prompted: headless sessions arrive with `ask_tools` already
        # emptied upstream, so an ask rule cannot force a prompt there.
        self._preapproved_tool_names -= self._ask_tool_names

    def pre_approvals_allowed_by_policy(self, tool_names: Iterable[str]) -> set[str]:
        """Reduce a pre-approval list, such as a flow config's, to what admin policy permits.

        Returns the names unchanged when the ceiling is off for this session.
        """
        names = set(tool_names)
        permitted = self._policy_permitted_preapprovals
        return names if permitted is None else names & permitted

    def ask_listed_tool_names(self, tool_names: Iterable[str]) -> set[str]:
        """Return the subset of ``tool_names`` an admin `ask` rule forces to prompt."""
        return {name for name in tool_names if name in self._ask_tool_names}

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

        This is a local check against the registry's pre-approved names, which come from privilege groups, capability
        tools, admin `allow` rules, and client-injected tools where a session opted in. Use this for routing decisions
        where you don't need per-call checks.
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
            # Rails hashes this string verbatim and compares it with the hash of the
            # client's `JSON.stringify(toolArgs)`, so keep the model's key order and raw UTF-8.
            tool_args_json = json.dumps(
                tool_args, separators=(",", ":"), ensure_ascii=False
            )
            sorted_tool_args_json = json.dumps(
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

        cache_key = (tool_name, sorted_tool_args_json)
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
                    log.debug(
                        "Tool call approved via session approval",
                        extra={
                            "tool_name": tool_name,
                            "workflow_id": self._workflow_id,
                        },
                    )
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

    def mcp_tool_names(self) -> list[str]:
        """Return the names of all MCP tools registered in this session.

        Callers that want to include MCP tools in a toolset must explicitly
        request them by passing the result of this method into ``toolset()``.
        MCP tools are no longer injected unconditionally by ``toolset()`` —
        each consumer opts in deliberately.

        Returns:
            A list of MCP tool name strings (may be empty when no MCP tools
            are connected to the session).
        """
        return list(self._mcp_tool_names)

    def toolset(
        self,
        tool_names: list[str],
        tool_options: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Toolset:
        """Create a Toolset instance representing complete collection of tools available to an agent.

        Args:
            tool_names: A list of tool names to include in the Toolset.
                MCP tools are NOT added automatically; callers that want them
                must include ``tools_registry.mcp_tool_names()`` in this list.
            tool_options: Optional dict mapping tool names to their option overrides.
                Example: {"create_merge_request_note": {"force_internal": True}}

        Returns:
            A new Toolset instance containing the requested tools.
        """

        # Only tools this toolset would otherwise include count as denied for
        # it; a deny rule for a tool the flow never had is not a block.
        denied_tools = {
            t
            for t in tool_names
            if t in self._denied_tools and t in self._enabled_tools
        }

        if denied_tools:
            tool_names = [t for t in tool_names if t not in denied_tools]

        all_tools = {}
        unavailable_tools: dict[str, UnavailableReason] = {}
        for tool_name in tool_names:
            if tool_name not in self._enabled_tools:
                unavailable_tools[tool_name] = self._unavailable_reason(tool_name)
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

        if unavailable_tools:
            _report_unavailable_tools(unavailable_tools)

        return Toolset(
            pre_approved=pre_approved,
            all_tools=all_tools,
            tool_options=tool_options or {},
            approval_policy=self,
            denied_tools=denied_tools,
        )

    def _unavailable_reason(self, tool_name: str) -> UnavailableReason:
        """Why a requested tool is not enabled. Governance-denied tools never get here.

        ``flow_restricted``: the tool's ``allowed_flows`` excludes this flow; no privilege unlocks
        it here. ``missing_privilege``: no privilege that grants the tool was
        enabled, typically a flow asking for more than its privileges in gitlab-org/gitlab allow.
        ``unsupported_client``:
        the privilege was enabled (or none is needed), but the client cannot run the tool.
        ``unknown``: the registry does not know the name, typically a typo in a custom flow.
        """
        grant = self._tool_grants.get(tool_name)
        if grant is None:
            if tool_name in self._capability_dependent_tool_names:
                return "unsupported_client"
            return "unknown"
        if (
            grant.allowed_flows is not None
            and self._registry_flow_id not in grant.allowed_flows
        ):
            return "flow_restricted"
        if not grant.privileges & self._enabled_privileges:
            return "missing_privilege"
        return "unsupported_client"

    # Built on the first unavailable tool, so a toolset that drops nothing pays nothing.
    @cached_property
    def _tool_grants(self) -> dict[str, _ToolGrant]:
        return _index_tool_grants()

    @cached_property
    def _capability_dependent_tool_names(self) -> set[str]:
        return {_tool_name(tool_cls) for tool_cls in _CAPABILITY_DEPENDENT_TOOLS}


def _report_unavailable_tools(
    unavailable_tools: dict[str, UnavailableReason],
) -> None:
    """Log the tools a toolset dropped, so a dropped tool is never silent.

    The structlog processors add ``workflow_id`` and the flow identity (``workflow_definition``,
    ``flow_name``), so neither is passed in.

    Field names are fixed (one list per reason) and tool names are values, never keys: tool
    names come from custom flow configs, and keys would add a log field per name.
    ``granting_privileges`` is the union of privileges that grant any ``missing_privilege_tools``
    entry; one of them per tool is enough. Empty fields are omitted.

    Info, not warning: dropping is also intended behaviour. Chat requests one tool list on every
    surface and lets the session's privileges trim it, so web chat drops the file and command
    tools on each compile. Filter on ``flow_restricted_tools``, ``missing_privilege_tools`` and
    ``unknown_tools`` to find real flow/privilege mismatches.
    """

    def tools_with(reason: UnavailableReason) -> list[str]:
        return sorted(name for name, r in unavailable_tools.items() if r == reason)

    missing_privilege_tools = tools_with("missing_privilege")
    fields = {
        "flow_restricted_tools": tools_with("flow_restricted"),
        "missing_privilege_tools": missing_privilege_tools,
        "unsupported_client_tools": tools_with("unsupported_client"),
        "unknown_tools": tools_with("unknown"),
        "granting_privileges": sorted(
            {
                privilege
                for name in missing_privilege_tools
                for privilege in agent_privileges_for_tool(name)
            }
        ),
    }
    log.info(
        "Requested tools unavailable in toolset",
        **{key: value for key, value in fields.items() if value},
    )


def _reject_unchecked_allowed_flows(
    tool_classes: Iterable[Union[Type[BaseTool], Type[BaseModel]]],
) -> None:
    """Raise if a tool enabled outside the privilege map sets ``allowed_flows``, which would be silently ignored."""
    for tool_cls in tool_classes:
        if getattr(tool_cls, "allowed_flows", None) is not None:
            raise RuntimeError(
                f"Tool {tool_cls.__name__} sets 'allowed_flows', which is only "
                "applied to tools granted by an agent privilege"
            )


_reject_unchecked_allowed_flows(
    [*_DEFAULT_TOOLS, *NO_OP_TOOLS, *_CAPABILITY_DEPENDENT_TOOLS]
)

# Reserve every name the registry can enable outside the privilege map, so a
# feature tool cannot shadow a default, no-op, or capability-dependent tool.
# Runs after every class in this module is defined: a feature components module
# may import modules that themselves import this one (for example the flow
# platform), which arrives here partially initialized during discovery.
_merge_feature_tools(
    _AGENT_PRIVILEGES,
    discover_feature_tools(),
    reserved=[*_DEFAULT_TOOLS, *NO_OP_TOOLS, *_CAPABILITY_DEPENDENT_TOOLS],
)
