import asyncio
from typing import Any, Optional, Type, TypedDict, Union

from gitlab_cloud_connector import CloudConnectorUser
from langchain.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service import tools
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.tools import Toolset, ToolType


class ToolMetadata(TypedDict):
    outbox: asyncio.Queue
    inbox: asyncio.Queue
    gitlab_client: GitlabHttpClient
    gitlab_host: str


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

_READ_ONLY_GITLAB_TOOLS: list[Type[BaseTool]] = [
    tools.ListIssues,
    tools.GetIssue,
    tools.GetLogsFromJob,
    tools.GetMergeRequest,
    tools.ListMergeRequestDiffs,
    tools.ListAllMergeRequestNotes,
    tools.GetPipelineErrorsForMergeRequest,
    tools.GetProject,
    tools.DocumentationSearch,
    tools.GroupProjectSearch,
    tools.IssueSearch,
    tools.MergeRequestSearch,
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
    tools.ListEpicNotes,
    tools.GetEpicNote,
    tools.GetCommit,
    tools.ListCommits,
    tools.GetCommitDiff,
    tools.GetCommitComments,
    tools.GetWorkflowContext,
    tools.CiLinter,
]

_AGENT_PRIVILEGES: dict[str, list[Type[BaseTool]]] = {
    "read_write_files": [
        tools.ReadFile,
        tools.WriteFile,
        tools.EditFile,
        tools.ListDir,
        tools.FindFiles,
        tools.Grep,
        # tools.Mkdir,
    ],
    "use_git": [
        tools.git.Command,
    ],
    "read_write_gitlab": [
        tools.CreateIssue,
        tools.UpdateIssue,
        tools.CreateIssueNote,
        tools.CreateMergeRequest,
        tools.CreateMergeRequestNote,
        tools.UpdateMergeRequest,
        tools.CreateEpic,
        tools.UpdateEpic,
        tools.CreateCommit,
        *_READ_ONLY_GITLAB_TOOLS,
    ],
    "read_only_gitlab": _READ_ONLY_GITLAB_TOOLS,
    "run_commands": [
        tools.RunCommand,
    ],
}


class ToolsRegistry:
    _enabled_tools: dict[str, Union[BaseTool, Type[BaseModel]]]
    _preapproved_tool_names: set[str]

    @classmethod
    async def configure(
        cls,
        workflow_config: dict[str, Any],
        gl_http_client: GitlabHttpClient,
        outbox: asyncio.Queue,
        inbox: asyncio.Queue,
        gitlab_host: str,
        additional_tools: Optional[list[Type[BaseTool]]] = None,
        user: Optional[CloudConnectorUser] = None,
    ):
        if not workflow_config:
            raise RuntimeError("Failed to find tools configuration for workflow")

        if "agent_privileges_names" not in workflow_config:
            raise RuntimeError(
                f"Failed to find tools configuration for workflow {workflow_config.get('id', 'None')}"
            )

        if not additional_tools:
            additional_tools = []

        agent_privileges = workflow_config.get("agent_privileges_names", [])
        preapproved_tools = workflow_config.get(
            "pre_approved_agent_privileges_names", []
        )
        tool_metadata = ToolMetadata(
            outbox=outbox,
            inbox=inbox,
            gitlab_client=gl_http_client,
            gitlab_host=gitlab_host,
        )

        return cls(
            enabled_tools=agent_privileges,
            preapproved_tools=preapproved_tools,
            tool_metadata=tool_metadata,
            additional_tools=additional_tools,
            user=user,
        )

    def __init__(
        self,
        enabled_tools: list[str],
        preapproved_tools: list[str],
        tool_metadata: ToolMetadata,
        additional_tools: Optional[list[Type[BaseTool]]] = None,
        user: Optional[CloudConnectorUser] = None,
    ):
        if not additional_tools:
            additional_tools = []

        # Create a dictionary of default and NO_OP tools
        default_tools: dict[str, Union[BaseTool, Type[BaseModel]]] = {
            **{tool_cls.tool_title: tool_cls for tool_cls in NO_OP_TOOLS},  # type: ignore
            **{tool.name: tool for tool in [tool_cls() for tool_cls in _DEFAULT_TOOLS]},
        }

        # Add additional tools separately
        additional_tool_dict = {tool.name: tool for tool in additional_tools}

        # Combine all tools
        self._enabled_tools = {
            **default_tools,
            **additional_tool_dict,
        }

        self._preapproved_tool_names = set(default_tools.keys())

        for privilege in enabled_tools:
            for tool_cls in _AGENT_PRIVILEGES[privilege]:
                tool = tool_cls(metadata=tool_metadata)

                # If user is passed, we check user permission to access this tool
                if user:
                    tool_primitive = getattr(tool, "unit_primitive", None)
                    if tool_primitive and not user.can(tool_primitive):
                        continue

                self._enabled_tools[tool.name] = tool
                if privilege in preapproved_tools:
                    self._preapproved_tool_names.add(tool.name)

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

    def approval_required(self, tool_name: str) -> bool:
        """Check if a tool requires human approval before execution.

        Args:
            tool_name: The name of the tool to check

        Returns:
            False if the tool is in the preapproved list,
            True otherwise.
        """
        return tool_name not in self._preapproved_tool_names

    def toolset(self, tool_names: list[str]) -> Toolset:
        """Create a Toolset instance representing complete collection of tools available to an agent.

        Args:
            tool_names: A list of tool names to include in the Toolset.

        Returns:
            A new Toolset instance containing the requested tools.
        """
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
