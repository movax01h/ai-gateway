import os
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, List

from dependency_injector.wiring import Provide, inject
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.tools_executor import ToolsExecutor
from duo_workflow_service.checkpointer.gitlab_workflow import WorkflowStatusEventEnum
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import (
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

MAX_TOKENS_TO_SAMPLE = 8192
DEBUG = os.getenv("DEBUG")
MAX_MESSAGE_LENGTH = 200
RECURSION_LIMIT = 500


class Routes(StrEnum):
    CONTINUE = "continue"
    NO_CONVERSATION_HISTORY = "no_conversation_history"
    SHOW_AGENT_MESSAGE = "show_agent_message"
    TOOL_USE = "tool_use"
    STOP = "stop"


CHAT_READ_ONLY_TOOLS = [
    "list_issues",
    "get_issue",
    "list_issue_notes",
    "get_issue_note",
    "get_job_logs",
    "get_merge_request",
    "get_pipeline_errors",
    "get_project",
    "run_read_only_git_command",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_merge_request_search",
    "read_file",
    "list_dir",
    "find_files",
    "grep",
    "get_epic",
    "list_epics",
    "scan_directory_tree",
    "list_epic_notes",
    "get_epic_note",
    "get_commit",
    "list_commits",
    "get_commit_comments",
    "get_commit_diff",
]


CHAT_GITLAB_MUTATION_TOOLS = [
    "create_issue",
    "update_issue",
    "create_issue_note",
    "create_merge_request",
    "update_merge_request",
    "create_merge_request_note",
    "create_epic",
    "update_epic",
    "create_commit",
]


CHAT_MUTATION_TOOLS = [
    "create_file_with_contents",
    "edit_file",
    # "mkdir",
]


class Workflow(AbstractWorkflow):
    _stream: bool = True
    _agent: ChatAgent

    def _are_tools_called(self, state: ChatWorkflowState) -> Routes:
        if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
            return Routes.STOP

        if state["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED:
            return Routes.STOP

        history: List[BaseMessage] = state["conversation_history"][self._agent.name]
        last_message = history[-1]
        if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
            return Routes.TOOL_USE

        return Routes.STOP

    def get_workflow_state(self, goal: str) -> ChatWorkflowState:
        contextElements = self._context_elements or []

        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            message_sub_type=None,
            content=f"Starting chat: {goal}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            context_elements=contextElements,
        )

        return ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={
                self._agent.name: [
                    HumanMessage(
                        content=goal,
                        additional_kwargs={
                            "additional_context": self._additional_context
                        },
                    ),
                ]
            },
            ui_chat_log=[initial_ui_chat_log],
            last_human_input=None,
            context_elements=contextElements,
            project=self._project,
            cancel_tool_message=None,
        )

    async def get_graph_input(self, goal: str, status_event: str) -> Any:
        match status_event:
            case WorkflowStatusEventEnum.START:
                return self.get_workflow_state(goal)
            case WorkflowStatusEventEnum.RESUME:
                state_update: dict[str, Any] = {"status": WorkflowStatusEnum.EXECUTION}
                next_step = "agent"

                match self._approval and self._approval.WhichOneof("user_decision"):
                    case "approval":
                        next_step = "run_tools"
                    case "rejection":
                        state_update["cancel_tool_message"] = self._approval.rejection.message  # type: ignore
                    case _:
                        state_update["conversation_history"] = {
                            self._agent.name: [
                                HumanMessage(
                                    content=goal,
                                    additional_kwargs={
                                        "additional_context": self._additional_context
                                    },
                                )
                            ]
                        }

                return Command(goto=next_step, update=state_update)
            case _:
                return None

    @inject
    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
        prompt_registry: LocalPromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
    ):
        self.log.info(
            "ChatWorkflow._compile: Starting chat workflow compilation",
            workflow_id=self._workflow_id,
            goal=goal,
        )

        self._goal = goal
        graph = StateGraph(ChatWorkflowState)
        tools = self._get_tools()
        agents_toolset = tools_registry.toolset(tools)

        self._agent: ChatAgent = prompt_registry.get(  # type: ignore[assignment]
            "chat/agent", tools=agents_toolset.bindable, prompt_version="^1.0.0"  # type: ignore[arg-type]
        )
        self._agent.tools_registry = tools_registry

        tools_runner = ToolsExecutor(
            tools_agent_name=self._agent.name,
            toolset=agents_toolset,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
        ).run

        graph.add_node("agent", self._agent.run)
        graph.add_node("run_tools", tools_runner)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            self._are_tools_called,
            {
                Routes.TOOL_USE: "run_tools",
                Routes.STOP: END,
            },
        )
        graph.add_edge("run_tools", "agent")

        return graph.compile(checkpointer=checkpointer)

    def _get_tools(self):
        available_tools = CHAT_READ_ONLY_TOOLS + CHAT_MUTATION_TOOLS

        if is_feature_enabled(FeatureFlag.DUO_WORKFLOW_WORKHORSE):
            available_tools += CHAT_GITLAB_MUTATION_TOOLS

        mcp_enabled = self._workflow_config.get("mcp_enabled", False)
        if is_feature_enabled(FeatureFlag.DUO_WORKFLOW_MCP_SUPPORT) and mcp_enabled:
            available_tools += [tool.name for tool in self._additional_tools]

        return available_tools

    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(error, extra={"workflow_id": self._workflow_id})
