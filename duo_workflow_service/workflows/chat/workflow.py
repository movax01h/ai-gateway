import os
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from duo_workflow_service.agents.agent import Agent
from duo_workflow_service.agents.prompts import CHAT_SYSTEM_PROMPT
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
from duo_workflow_service.interceptors.feature_flag_interceptor import (
    current_feature_flag_context,
)
from duo_workflow_service.llm_factory import new_chat_client
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow

MAX_TOKENS_TO_SAMPLE = 8192
DEBUG = os.getenv("DEBUG")
MAX_MESSAGE_LENGTH = 200
RECURSION_LIMIT = 500
AGENT_NAME = "Chat Agent"


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


CHAT_MUTATION_TOOLS = [
    "create_file_with_contents",
    "edit_file",
    "mkdir",
]


class Workflow(AbstractWorkflow):
    _stream: bool = True

    def _are_tools_called(self, state: ChatWorkflowState) -> Routes:
        if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
            return Routes.STOP

        history: List[BaseMessage] = state["conversation_history"][AGENT_NAME]
        last_message = history[-1]
        if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
            return Routes.TOOL_USE

        return Routes.STOP

    async def _execute_agent(self, state: ChatWorkflowState) -> Dict[str, Any]:
        # First run the agent
        agent_result = await self._agent.run(state)
        contextElements = self._context_elements or []

        history: List[BaseMessage] = agent_result["conversation_history"][AGENT_NAME]
        if not history:
            return {"status": WorkflowStatusEnum.EXECUTION}
        last_message = history[-1]

        # Combine the agent result with the UI message
        result = {
            **agent_result,
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
        }

        if not isinstance(last_message, AIMessage) or len(last_message.tool_calls) == 0:
            result["ui_chat_log"] = [
                UiChatLog(
                    message_type=MessageTypeEnum.AGENT,
                    content=StrOutputParser().invoke(last_message) or "",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=None,
                    context_elements=contextElements,
                )
            ]

        return result

    def get_workflow_state(self, goal: str) -> ChatWorkflowState:
        contextElements = self._context_elements or []

        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Starting chat: {goal}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            context_elements=contextElements,
        )

        system_prompt = CHAT_SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            project_id=self._project.get("id", "unknown"),
            project_name=self._project.get("name", "unknown"),
            project_url=self._project.get("web_url", "unknown"),
        )

        return ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={
                AGENT_NAME: [
                    SystemMessage(content=system_prompt),
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
        )

    async def get_graph_input(self, goal: str, status_event: str) -> Any:
        match status_event:
            case WorkflowStatusEventEnum.START:
                return self.get_workflow_state(goal)
            case WorkflowStatusEventEnum.RESUME:
                return Command(
                    goto="agent",
                    update={
                        "status": WorkflowStatusEnum.EXECUTION,
                        "conversation_history": {
                            AGENT_NAME: [
                                HumanMessage(
                                    content=goal,
                                    additional_kwargs={
                                        "additional_context": self._additional_context
                                    },
                                )
                            ]
                        },
                    },
                )
            case _:
                return None

    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
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

        tools_runner = ToolsExecutor(
            tools_agent_name=AGENT_NAME,
            toolset=agents_toolset,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
        ).run

        self._agent = Agent(
            goal="",
            system_prompt="",
            name=AGENT_NAME,
            model=new_chat_client(max_tokens=MAX_TOKENS_TO_SAMPLE),
            toolset=agents_toolset,
            workflow_id=self._workflow_id,
            http_client=self._http_client,
            workflow_type=self._workflow_type,
            check_events=False,
        )

        graph.add_node("agent", self._execute_agent)
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

    def log_workflow_elements(self, element):
        self.log.info("###############################")
        if "ui_chat_log" in element:
            for log in element["ui_chat_log"]:
                self.log.info(
                    f"%s: %{'' if DEBUG else f'.{MAX_MESSAGE_LENGTH}'}s",
                    log["message_type"],
                    log["content"],
                )

    def _get_tools(self):
        available_tools = CHAT_READ_ONLY_TOOLS
        feature_flags = current_feature_flag_context.get()
        if "duo_workflow_chat_mutation_tools" in feature_flags:
            available_tools = CHAT_READ_ONLY_TOOLS + CHAT_MUTATION_TOOLS

        if "duo_workflow_mcp_support" in feature_flags:
            available_tools += [tool.name for tool in self._additional_tools]

        return available_tools

    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(error, extra={"workflow_id": self._workflow_id})
