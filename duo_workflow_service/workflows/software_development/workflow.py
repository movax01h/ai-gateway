import os
from datetime import datetime, timezone
from enum import StrEnum
from functools import partial
from typing import Annotated

# pylint disables are going to be fixed via
# https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-service/-/issues/78
from langchain_core.messages import (  # pylint: disable=no-langgraph-langchain-imports
    AIMessage,
)
from langgraph.checkpoint.memory import (  # pylint: disable=no-langgraph-langchain-imports
    BaseCheckpointSaver,
)
from langgraph.graph import (  # pylint: disable=no-langgraph-langchain-imports
    END,
    StateGraph,
)

from duo_workflow_service.agents import (
    Agent,
    HandoverAgent,
    HumanApprovalCheckExecutor,
    HumanApprovalEntryExecutor,
    PlanSupervisorAgent,
    PlanTerminatorAgent,
    ToolsExecutor,
)
from duo_workflow_service.agents.prompts import (
    BUILD_CONTEXT_SYSTEM_MESSAGE,
    EXECUTOR_SYSTEM_MESSAGE,
    HANDOVER_TOOL_NAME,
    PLANNER_GOAL,
    PLANNER_PROMPT,
    SET_TASK_STATUS_TOOL_NAME,
)
from duo_workflow_service.components import GoalDisambiguationComponent, ToolsRegistry
from duo_workflow_service.entities import (
    MessageTypeEnum,
    Plan,
    ToolStatus,
    UiChatLog,
    WorkflowEventType,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.llm_factory import new_chat_client
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow

# Constants
QUEUE_MAX_SIZE = 1
MAX_TOKENS_TO_SAMPLE = 4096
RECURSION_LIMIT = 300
DEBUG = os.getenv("DEBUG")
MAX_MESSAGES_TO_DISPLAY = 5
MAX_MESSAGE_LENGTH = 200

EXECUTOR_TOOLS = [
    "create_issue",
    "list_issues",
    "get_issue",
    "update_issue",
    "create_issue_note",
    "create_merge_request_note",
    "list_issue_notes",
    "get_issue_note",
    "create_merge_request",
    "get_job_logs",
    "get_merge_request",
    "get_pipeline_errors",
    "get_project",
    "run_read_only_git_command",
    "run_git_command",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_merge_request_search",
    "run_command",
    "read_file",
    "ls_files",
    "create_file_with_contents",
    "edit_file",
    "find_files",
    "grep_files",
    "mkdir",
    "add_new_task",
    "remove_task",
    "update_task_description",
    "get_plan",
    "set_task_status",
    "handover_tool",
    "get_epic",
    "list_epics",
    "create_epic",
    "update_epic",
]

CONTEXT_BUILDER_TOOLS = [
    "list_issues",
    "get_issue",
    "list_issue_notes",
    "get_issue_note",
    "get_job_logs",
    "get_merge_request",
    "get_project",
    "get_pipeline_errors",
    "run_read_only_git_command",
    "run_git_command",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_merge_request_search",
    "read_file",
    "ls_files",
    "find_files",
    "grep_files",
    "run_command",
    "handover_tool",
    "get_epic",
    "list_epics",
]

PLANNER_TOOLS = [
    "get_plan",
    "add_new_task",
    "remove_task",
    "update_task_description",
    "handover_tool",
]


class Routes(StrEnum):
    END = "end"
    CALL_TOOL = "call_tool"
    SUPERVISOR = PlanSupervisorAgent.__name__
    HANDOVER = HandoverAgent.__name__
    BUILD_CONTEXT = "build_context"
    STOP = "stop"
    CHAT = "chat"
    WAIT_FOR_HUMAN_INPUT = "wait_for_human_input"


def _router(
    routed_agent_name: str,
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    last_message = state["conversation_history"][routed_agent_name][-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        if last_message.tool_calls[0]["name"] == HANDOVER_TOOL_NAME:
            return Routes.HANDOVER
        return Routes.CALL_TOOL

    return Routes.SUPERVISOR


def _approval_router(
    state: WorkflowState,
) -> Routes:
    if not os.environ.get("WORKFLOW_INTERRUPT", False) or os.getenv("USE_MEMSAVER"):
        return Routes.HANDOVER

    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    event = state.get("last_human_input", None)
    if event:
        if event["event_type"] == WorkflowEventType.RESUME:
            return Routes.HANDOVER
        elif event["event_type"] == WorkflowEventType.STOP:
            return Routes.STOP
    return Routes.CHAT


def _should_continue(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.ERROR, WorkflowStatusEnum.CANCELLED]:
        return Routes.STOP

    return Routes.BUILD_CONTEXT


class Workflow(AbstractWorkflow):
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph, graph_config
    ):
        log_exception(error, extra={"workflow_id": self._workflow_id})

    def _setup_executor(
        self, goal: str, tools_registry: ToolsRegistry, base_model_executor
    ):
        executor = Agent(
            goal=goal,
            model=base_model_executor,
            name="executor",
            system_prompt=EXECUTOR_SYSTEM_MESSAGE.format(
                set_task_status_tool_name=SET_TASK_STATUS_TOOL_NAME,
                handover_tool_name=HANDOVER_TOOL_NAME,
                get_plan_tool_name=tools_registry.get("get_plan").name,  # type: ignore
                project_id=self._project["id"],
                project_name=self._project["name"],
                project_url=self._project["http_url_to_repo"],
            ),
            tools=tools_registry.get_batch(EXECUTOR_TOOLS),
            workflow_id=self._workflow_id,
            http_client=self._http_client,
        )

        return {
            "agent": executor,
            "tools": EXECUTOR_TOOLS,
            "supervisor": PlanSupervisorAgent(supervised_agent_name=executor.name),
            "handover": HandoverAgent(
                new_status=WorkflowStatusEnum.COMPLETED,
                handover_from=executor.name,
                include_conversation_history=True,
            ),
            "tools_executor": ToolsExecutor(
                tools_agent_name="executor",
                agent_tools=tools_registry.get_handlers(EXECUTOR_TOOLS),
                workflow_id=self._workflow_id,
            ),
        }

    def _setup_planner(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        base_model_planner,
        executor_tools,
    ):
        planner = Agent(
            goal=PLANNER_GOAL.format(
                executor_agent_prompt=EXECUTOR_SYSTEM_MESSAGE,
                handover_tool_name=HANDOVER_TOOL_NAME,
                executor_agent_tools="\n".join(
                    [
                        f"{t.name}: {t.description}"
                        for t in tools_registry.get_handlers(executor_tools)
                    ]
                ),
                goal=goal,
                get_plan_tool_name=tools_registry.get("get_plan").name,  # type: ignore
                add_new_task_tool_name=tools_registry.get("add_new_task").name,  # type: ignore
                remove_task_tool_name=tools_registry.get("remove_task").name,  # type: ignore
                update_task_description_tool_name=tools_registry.get("update_task_description").name,  # type: ignore
                project_id=self._project["id"],
                project_name=self._project["name"],
                project_url=self._project["http_url_to_repo"],
            ),
            model=base_model_planner,
            name="planner",
            workflow_id=self._workflow_id,
            http_client=self._http_client,
            system_prompt=PLANNER_PROMPT,
            tools=tools_registry.get_batch(PLANNER_TOOLS),
        )

        return {
            "agent": planner,
            "tools": PLANNER_TOOLS,
            "supervisor": PlanSupervisorAgent(supervised_agent_name="planner"),
            "tools_executor": ToolsExecutor(
                tools_agent_name="planner",
                agent_tools=[],
                workflow_id=self._workflow_id,
            ),
        }

    def _setup_approval_for_stage(self, stage_name):
        return {
            "planning_approval_entry": HumanApprovalEntryExecutor(
                stage_name,
                self._workflow_id,
            ),
            "planning_approval_check": HumanApprovalCheckExecutor(
                stage_name,
                self._workflow_id,
            ),
        }

    def _setup_workflow_graph(
        self,
        graph: StateGraph,
        executor_components,
        planner_components,
        planner_approval_components,
        tools_registry,
        goal,
    ):
        # Add nodes to the graph
        graph.set_entry_point("build_context")

        last_node_name = self._add_context_builder_nodes(graph, goal, tools_registry)
        disambiguation_component = GoalDisambiguationComponent(
            goal=goal,
            model=new_chat_client(),
            http_client=self._http_client,
            workflow_id=self._workflow_id,
            tools_registry=tools_registry,
        )
        disambiguation_entry_node = disambiguation_component.attach(
            graph=graph,
            component_exit_node="planning",
            graph_termination_node="plan_terminator",
            component_execution_state=WorkflowStatusEnum.PLANNING,
        )

        graph.add_edge(last_node_name, disambiguation_entry_node)
        # graph.add_edge(disambiguation_exit_node, "planning")
        graph.add_node("planning", planner_components["agent"].run)
        graph.add_node("update_plan", planner_components["tools_executor"].run)
        graph.add_node("planning_supervisor", planner_components["supervisor"].run)
        graph.add_node(
            "planning_approval_entry",
            planner_approval_components["planning_approval_entry"].run,
        )
        graph.add_node(
            "planning_approval_check",
            planner_approval_components["planning_approval_check"].run,
        )
        graph.add_edge("update_plan", "planning")
        graph.add_edge("planning_supervisor", "planning")

        graph.add_conditional_edges(
            "planning",
            partial(_router, "planner"),
            {
                Routes.CALL_TOOL: "update_plan",
                Routes.SUPERVISOR: "planning_supervisor",
                Routes.HANDOVER: "planning_approval_entry",
                Routes.STOP: "plan_terminator",
            },
        )
        graph.add_edge("planning_approval_entry", "planning_approval_check")

        graph.add_conditional_edges(
            "planning_approval_check",
            _approval_router,
            {
                Routes.HANDOVER: "set_status_to_execution",
                Routes.STOP: "plan_terminator",
                Routes.CHAT: "planning",
            },
        )

        plan_terminator = PlanTerminatorAgent(workflow_id=self._workflow_id)
        graph.add_node("plan_terminator", plan_terminator.run)

        graph.add_node(
            "set_status_to_execution",
            HandoverAgent(
                new_status=WorkflowStatusEnum.EXECUTION,
                handover_from=planner_components["agent"].name,
            ).run,
        )
        graph.add_node("execution", executor_components["agent"].run)

        graph.add_node("execution_tools", executor_components["tools_executor"].run)
        graph.add_node("execution_supervisor", executor_components["supervisor"].run)
        graph.add_node("execution_handover", executor_components["handover"].run)

        graph.add_edge("set_status_to_execution", "execution")
        graph.add_edge("execution_supervisor", "execution")
        graph.add_edge("execution_tools", "execution")
        graph.add_edge("execution_handover", END)
        graph.add_conditional_edges(
            "execution",
            partial(_router, "executor"),
            {
                Routes.CALL_TOOL: "execution_tools",
                Routes.HANDOVER: "execution_handover",
                Routes.SUPERVISOR: "execution_supervisor",
                Routes.STOP: "plan_terminator",
            },
        )

        graph.add_edge("plan_terminator", END)

        return graph

    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
    ):
        base_model_planner = new_chat_client()
        base_model_executor = new_chat_client(max_tokens=MAX_TOKENS_TO_SAMPLE)

        graph = StateGraph(WorkflowState)

        executor_components = self._setup_executor(
            goal, tools_registry, base_model_executor
        )
        planner_components = self._setup_planner(
            goal, tools_registry, base_model_planner, executor_components["tools"]
        )
        planner_approval_components = self._setup_approval_for_stage(
            planner_components["agent"].name
        )

        graph = self._setup_workflow_graph(
            graph,
            executor_components,
            planner_components,
            planner_approval_components,
            tools_registry,
            goal,
        )

        return graph.compile(checkpointer=checkpointer)

    def get_workflow_state(self, goal: str) -> WorkflowState:
        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Starting workflow with goal: {goal}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
        )

        return WorkflowState(
            plan=Plan(steps=[]),
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            last_human_input=None,
            handover=[],
            ui_chat_log=[initial_ui_chat_log],
        )

    def log_workflow_elements(self, element):
        if "conversation_history" in element:
            for agent, messages in element["conversation_history"].items():
                self.log.info("agent: %s", agent)
                messages = messages if DEBUG else messages[-MAX_MESSAGES_TO_DISPLAY:]
                for message in messages:
                    self.log.info(
                        f"%s: %{'' if DEBUG else f'.{MAX_MESSAGE_LENGTH}'}s",
                        message.__class__.__name__,
                        message.content,
                    )
                self.log.info("--------------------------")
                if "status" in element:
                    self.log.info(element["status"])
                if "plan" in element:
                    self.log.info(element["plan"])
            self.log.info("###############################")

    def _setup_context_builder(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
    ):
        context_builder = Agent(
            goal=goal,
            model=new_chat_client(),  # type: ignore
            name="context_builder",
            system_prompt=BUILD_CONTEXT_SYSTEM_MESSAGE.format(
                handover_tool_name=HANDOVER_TOOL_NAME,
                project_id=self._project["id"],
                project_name=self._project["name"],
                project_url=self._project["http_url_to_repo"],
            ),
            tools=tools_registry.get_batch(CONTEXT_BUILDER_TOOLS),
            workflow_id=self._workflow_id,
            http_client=self._http_client,
        )

        return {
            "agent": context_builder,
            "tools": CONTEXT_BUILDER_TOOLS,
            "handover": HandoverAgent(
                new_status=WorkflowStatusEnum.PLANNING,
                handover_from=context_builder.name,
                include_conversation_history=True,
            ),
            "supervisor": PlanSupervisorAgent(supervised_agent_name="context_builder"),
            "tools_executor": ToolsExecutor(
                tools_agent_name=context_builder.name,
                agent_tools=tools_registry.get_handlers(CONTEXT_BUILDER_TOOLS),
                workflow_id=self._workflow_id,
            ),
        }

    def _add_context_builder_nodes(
        self, graph: StateGraph, goal: str, tools_registry: ToolsRegistry
    ) -> Annotated[str, "The name of the last handover node"]:
        context_builder_components = self._setup_context_builder(goal, tools_registry)

        graph.add_node("build_context", context_builder_components["agent"].run)
        graph.add_node(
            "build_context_tools", context_builder_components["tools_executor"].run
        )
        graph.add_node(
            "build_context_handover", context_builder_components["handover"].run
        )
        graph.add_node(
            "build_context_supervisor", context_builder_components["supervisor"].run
        )

        graph.add_conditional_edges(
            "build_context",
            partial(_router, "context_builder"),
            {
                Routes.CALL_TOOL: "build_context_tools",
                Routes.HANDOVER: "build_context_handover",
                Routes.SUPERVISOR: "build_context_supervisor",
                Routes.STOP: "plan_terminator",
            },
        )
        graph.add_conditional_edges(
            "build_context_tools",
            _should_continue,
            {
                Routes.BUILD_CONTEXT: "build_context",
                Routes.STOP: "plan_terminator",
            },
        )

        graph.add_edge("build_context_supervisor", "build_context")
        return "build_context_handover"
