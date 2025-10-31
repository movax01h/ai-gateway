# pylint: disable=attribute-defined-outside-init
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional, override

from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.registry import LocalPromptRegistry
from contract import contract_pb2
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.chat_agent_factory import create_agent
from duo_workflow_service.agents.tools_executor import ToolsExecutor
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import (
    ApprovalStateRejection,
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    InvocationMetadata,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum


class Routes(StrEnum):
    CONTINUE = "continue"
    NO_CONVERSATION_HISTORY = "no_conversation_history"
    SHOW_AGENT_MESSAGE = "show_agent_message"
    TOOL_USE = "tool_use"
    STOP = "stop"


CHAT_READ_ONLY_TOOLS = [
    "get_job_logs",
    "get_merge_request",
    "get_pipeline_errors",
    "get_project",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_blob_search",
    "gitlab_merge_request_search",
    "gitlab_documentation_search",
    "read_file",
    "read_files",
    "get_repository_file",
    "list_dir",
    "find_files",
    "grep",
    "list_repository_tree",
    "get_commit",
    "list_commits",
    "get_commit_diff",
    "get_work_item",
    "get_work_item_notes",
    "list_work_items",
    "list_vulnerabilities",
    "get_current_user",
    "get_vulnerability_details",
]


CHAT_GITLAB_MUTATION_TOOLS = [
    "update_vulnerability_severity",
    "create_merge_request",
    "update_merge_request",
    "create_merge_request_note",
    "create_commit",
    "dismiss_vulnerability",
    "confirm_vulnerability",
    "create_work_item",
    "create_work_item_note",
    "link_vulnerability_to_issue",
    "update_work_item",
    "revert_to_detected_vulnerability",
    "create_vulnerability_issue",
]


CHAT_MUTATION_TOOLS = [
    "create_file_with_contents",
    "edit_file",
    "mkdir",
]

RUN_COMMAND_TOOLS = ["run_command"]

GIT_TOOLS = ["run_git_command"]


class Workflow(AbstractWorkflow):
    _stream: bool = True
    _agent: ChatAgent
    _tools_override: list[str]
    _workflow_id: str
    _workflow_type: CategoryEnum

    # pylint: disable=dangerous-default-value
    @inject
    def __init__(
        self,
        workflow_id: str,
        workflow_metadata: Dict[str, Any],
        workflow_type: CategoryEnum,
        invocation_metadata: InvocationMetadata = {
            "base_url": "",
            "gitlab_token": "",
        },
        mcp_tools: list[contract_pb2.McpTool] = [],
        user: Optional[CloudConnectorUser] = None,
        additional_context: Optional[list[AdditionalContext]] = None,
        approval: Optional[contract_pb2.Approval] = None,
        system_template_override: str | None = None,
        prompt_registry: LocalPromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
        **kwargs,
    ):
        self._tools_override = kwargs.pop("tools_override", None)
        self.system_template_override = system_template_override
        self._workflow_id = workflow_id
        self._workflow_type = workflow_type

        super().__init__(
            workflow_id=workflow_id,
            workflow_metadata=workflow_metadata,
            workflow_type=workflow_type,
            invocation_metadata=invocation_metadata,
            mcp_tools=mcp_tools,
            user=user,
            additional_context=additional_context,
            approval=approval,
            prompt_registry=prompt_registry,  # type: ignore[arg-type]
            internal_event_client=internal_event_client,
            **kwargs,
        )

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
        initial_ui_chat_log = UiChatLog(
            message_sub_type=None,
            message_type=MessageTypeEnum.USER,
            content=goal,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            additional_context=self._additional_context,
        )

        conversation_history: List[BaseMessage] = []

        conversation_history.append(
            HumanMessage(
                content=goal,
                additional_kwargs={"additional_context": self._additional_context},
            ),
        )

        return ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={self._agent.name: conversation_history},
            ui_chat_log=[initial_ui_chat_log],
            last_human_input=None,
            goal=goal,
            project=self._project,
            namespace=self._namespace,
            approval=None,
            preapproved_tools=self._preapproved_tools,
        )

    async def get_graph_input(
        self, goal: str, status_event: str, checkpoint_tuple: Optional[CheckpointTuple]
    ) -> Any:
        new_chat_message = goal

        match status_event:
            case WorkflowStatusEventEnum.START:
                return self.get_workflow_state(goal)
            case _:
                state_update: dict[str, Any] = {
                    "status": WorkflowStatusEnum.EXECUTION,
                    "preapproved_tools": self._preapproved_tools,
                }
                next_step = "agent"

                match self._approval and self._approval.WhichOneof("user_decision"):
                    case "approval":
                        next_step = "run_tools"
                    case "rejection":
                        new_chat_message = self._approval.rejection.message  # type: ignore
                        state_update["approval"] = ApprovalStateRejection(
                            message=new_chat_message
                        )
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

                if new_chat_message and new_chat_message != "null":
                    new_message_chat_log = UiChatLog(
                        message_type=MessageTypeEnum.USER,
                        message_sub_type=None,
                        content=new_chat_message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=None,
                        additional_context=self._additional_context,
                    )
                    state_update["ui_chat_log"] = [new_message_chat_log]

                return Command(goto=next_step, update=state_update)

    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
    ):
        self.log.info(
            "ChatWorkflow._compile: Starting chat workflow compilation",
            workflow_id=self._workflow_id,
        )

        self._goal = goal
        graph = StateGraph(ChatWorkflowState)

        if self._tools_override is not None:
            tools = self._tools_override
        else:
            tools = self._get_tools()

        agents_toolset = tools_registry.toolset(tools)

        self._agent: ChatAgent = create_agent(
            user=self._user,
            tools_registry=tools_registry,
            internal_event_category=__name__,
            tools=agents_toolset,
            prompt_registry=self._prompt_registry,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            system_template_override=self.system_template_override,
        )

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
        available_tools = (
            CHAT_READ_ONLY_TOOLS
            + CHAT_MUTATION_TOOLS
            + RUN_COMMAND_TOOLS
            + GIT_TOOLS
            + CHAT_GITLAB_MUTATION_TOOLS
        )

        return available_tools

    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(
            error, extra={"workflow_id": self._workflow_id, "source": __name__}
        )

    @override
    def _support_namespace_level_workflow(self) -> bool:
        return True
