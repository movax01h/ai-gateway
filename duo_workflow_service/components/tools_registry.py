import asyncio
from typing import Optional, Type, TypedDict, Union

from langchain.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service import tools
from duo_workflow_service.gitlab.http_client import GitlabHttpClient

ToolType = Union[BaseTool, Type[BaseModel]]


class ToolMetadata(TypedDict):
    outbox: asyncio.Queue
    inbox: asyncio.Queue
    gitlab_client: GitlabHttpClient
    gitlab_host: str


# This tools agent uses to interact with its internal state, they are required for
# a workflow to progress, and they do not pose any security risk, therefore they
# are being exempted from dynamic configuration.
_DEFAULT_TOOLS: list[Type[BaseTool]] = [
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
]

_AGENT_PRIVILEGES: dict[str, list[Type[BaseTool]]] = {
    "read_write_files": [
        tools.ReadFile,
        tools.WriteFile,
        tools.EditFile,
        tools.LsFiles,
        tools.FindFiles,
        tools.Grep,
        tools.Mkdir,
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
        *_READ_ONLY_GITLAB_TOOLS,
    ],
    "read_only_gitlab": _READ_ONLY_GITLAB_TOOLS,
    "run_commands": [
        tools.RunCommand,
        tools.ReadOnlyGit,
    ],
}


class ToolsRegistry:
    _approved_tools: dict[str, Union[BaseTool, Type[BaseModel]]]

    @classmethod
    async def configure(
        cls,
        workflow_id: str,
        gl_http_client: GitlabHttpClient,
        outbox: asyncio.Queue,
        inbox: asyncio.Queue,
        *,
        gitlab_host: str,
    ):
        config = await gl_http_client.aget(
            f"/api/v4/ai/duo_workflows/workflows/{workflow_id}"
        )
        if not config or "agent_privileges_names" not in config:
            raise RuntimeError(
                f"Failed to fetch tools configuration for workflow {workflow_id}"
            )

        return cls(
            outbox=outbox,
            inbox=inbox,
            gl_http_client=gl_http_client,
            tools_configuration=config["agent_privileges_names"],
            gitlab_host=gitlab_host,
        )

    def __init__(
        self,
        outbox: asyncio.Queue,
        inbox: asyncio.Queue,
        gl_http_client: GitlabHttpClient,
        tools_configuration: list[str],
        *,
        gitlab_host: str,
    ):
        tool_metadata = ToolMetadata(
            outbox=outbox,
            inbox=inbox,
            gitlab_client=gl_http_client,
            gitlab_host=gitlab_host,
        )
        self._approved_tools = {
            **{tool_cls.tool_title: tool_cls for tool_cls in NO_OP_TOOLS},  # type: ignore
            **{tool.name: tool for tool in [tool_cls() for tool_cls in _DEFAULT_TOOLS]},
        }

        for privilege in tools_configuration:
            for tool_cls in _AGENT_PRIVILEGES[privilege]:
                tool = tool_cls(metadata=tool_metadata)
                self._approved_tools[tool.name] = tool

    def get(self, tool_name: str) -> Optional[ToolType]:
        return self._approved_tools.get(tool_name)

    def get_batch(self, tool_names: list[str]) -> list[ToolType]:
        return [
            self._approved_tools[tool_name]
            for tool_name in tool_names
            if tool_name in self._approved_tools
        ]

    def get_handlers(self, tool_names: list[str]) -> list[BaseTool]:
        tool_handlers: list[BaseTool] = []
        for tool_name in tool_names:
            handler = self._approved_tools.get(tool_name)
            if isinstance(handler, BaseTool):
                tool_handlers.append(handler)

        return tool_handlers
