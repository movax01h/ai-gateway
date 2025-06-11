import os
from datetime import datetime, timezone
from enum import StrEnum
from functools import partial
from typing import Annotated, Union

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
    PlanSupervisorAgent,
    PlanTerminatorAgent,
    ToolsExecutor,
)
from duo_workflow_service.agents.prompts import (
    BATCH_PLANNER_GOAL,
    BUILD_CONTEXT_SYSTEM_MESSAGE,
    EXECUTOR_SYSTEM_MESSAGE,
    HANDOVER_TOOL_NAME,
    PLANNER_GOAL,
    PLANNER_INSTRUCTIONS,
    PLANNER_PROMPT,
    PLANNER_TASK_BATCH_INSTRUCTIONS,
    SET_TASK_STATUS_TOOL_NAME,
)
from duo_workflow_service.components import (
    GoalDisambiguationComponent,
    PlanApprovalComponent,
    ToolsApprovalComponent,
    ToolsRegistry,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    Plan,
    ToolStatus,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.interceptors.feature_flag_interceptor import (
    current_feature_flag_context,
)
from duo_workflow_service.llm_factory import (
    AnthropicConfig,
    VertexConfig,
    create_chat_model,
)
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from duo_workflow_service.workflows.model_selection_utils import (
    get_sonnet_4_config_with_feature_flag,
)

# Constants
QUEUE_MAX_SIZE = 1
MAX_TOKENS_TO_SAMPLE = 8192
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
    "run_git_command",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_merge_request_search",
    "run_command",
    "read_file",
    "create_file_with_contents",
    "edit_file",
    "find_files",
    "grep",
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
    "get_repository_file",
    "list_dir",
    "list_epic_notes",
    "get_epic_note",
    "get_commit",
    "list_commits",
    "get_commit_comments",
    "get_commit_diff",
]

CONTEXT_BUILDER_TOOLS = [
    "get_previous_workflow_context",
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
    "find_files",
    "list_dir",
    "grep",
    "handover_tool",
    "get_epic",
    "list_epics",
    "get_repository_file",
    "list_epic_notes",
    "get_epic_note",
    "get_commit",
    "list_commits",
    "get_commit_comments",
    "get_commit_diff",
]

PLANNER_TOOLS = [
    "get_previous_workflow_context",
    "get_plan",
    "add_new_task",
    "remove_task",
    "update_task_description",
    "handover_tool",
]


class Routes(StrEnum):
    END = "end"
    CALL_TOOL = "call_tool"
    TOOLS_APPROVAL = "tools_approval"
    SUPERVISOR = PlanSupervisorAgent.__name__
    HANDOVER = HandoverAgent.__name__
    BUILD_CONTEXT = "build_context"
    STOP = "stop"
    CHAT = "chat"
    WAIT_FOR_HUMAN_INPUT = "wait_for_human_input"


def _router(
    routed_agent_name: str,
    tool_registry: ToolsRegistry,
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    last_message = state["conversation_history"][routed_agent_name][-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        if last_message.tool_calls[0]["name"] == HANDOVER_TOOL_NAME:
            return Routes.HANDOVER
        if any(
            tool_registry.approval_required(call["name"])
            for call in last_message.tool_calls
        ):
            return Routes.TOOLS_APPROVAL
        return Routes.CALL_TOOL

    return Routes.SUPERVISOR


def _should_continue(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.ERROR, WorkflowStatusEnum.CANCELLED]:
        return Routes.STOP

    return Routes.BUILD_CONTEXT


class Workflow(AbstractWorkflow):
    def _get_model_config(self) -> Union[AnthropicConfig, VertexConfig]:
        """Override to use Sonnet 4 model for software development tasks."""
        config = get_sonnet_4_config_with_feature_flag(self._workflow_type)
        return config if config else super()._get_model_config()

    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph, graph_config
    ):
        log_exception(error, extra={"workflow_id": self._workflow_id})

    def _setup_executor(
        self, goal: str, tools_registry: ToolsRegistry, base_model_executor
    ):
        executors_toolset = tools_registry.toolset(EXECUTOR_TOOLS)
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
            toolset=executors_toolset,
            workflow_id=self._workflow_id,
            http_client=self._http_client,
            workflow_type=self._workflow_type,
        )

        return {
            "agent": executor,
            "toolset": executors_toolset,
            "supervisor": PlanSupervisorAgent(supervised_agent_name=executor.name),
            "handover": HandoverAgent(
                new_status=WorkflowStatusEnum.COMPLETED,
                handover_from=executor.name,
                include_conversation_history=True,
            ),
            "tools_executor": ToolsExecutor(
                tools_agent_name="executor",
                toolset=executors_toolset,
                workflow_id=self._workflow_id,
                workflow_type=self._workflow_type,
            ),
        }

    def _setup_planner(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        base_model_planner,
        executor_toolset,
    ):
        available_feature_flags = current_feature_flag_context.get()
        if "batch_duo_workflow_planner_tasks" in available_feature_flags:
            planner_tools = PLANNER_TOOLS + ["create_plan"]
            planner_toolset = tools_registry.toolset(planner_tools)
            planner = Agent(
                goal=BATCH_PLANNER_GOAL.format(
                    executor_agent_prompt=EXECUTOR_SYSTEM_MESSAGE,
                    handover_tool_name=HANDOVER_TOOL_NAME,
                    executor_agent_tools="\n".join(
                        [
                            f"{tool_name}: {tool.description}"
                            for tool_name, tool in executor_toolset.items()
                        ]
                    ),
                    goal=goal,
                    create_plan_tool_name=tools_registry.get("create_plan").name,  # type: ignore
                    get_plan_tool_name=tools_registry.get("get_plan").name,  # type: ignore
                    add_new_task_tool_name=tools_registry.get("add_new_task").name,  # type: ignore
                    remove_task_tool_name=tools_registry.get("remove_task").name,  # type: ignore
                    update_task_description_tool_name=tools_registry.get("update_task_description").name,  # type: ignore
                    planner_instructions=self.planner_instructions(tools_registry),
                ),
                model=base_model_planner,
                name="planner",
                workflow_id=self._workflow_id,
                http_client=self._http_client,
                system_prompt=PLANNER_PROMPT,
                toolset=planner_toolset,
                workflow_type=self._workflow_type,
            )
        else:
            planner_tools = PLANNER_TOOLS
            planner_toolset = tools_registry.toolset(PLANNER_TOOLS)
            planner = Agent(
                goal=PLANNER_GOAL.format(
                    executor_agent_prompt=EXECUTOR_SYSTEM_MESSAGE,
                    handover_tool_name=HANDOVER_TOOL_NAME,
                    executor_agent_tools="\n".join(
                        [
                            f"{tool_name}: {tool.description}"
                            for tool_name, tool in executor_toolset.items()
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
                    planner_instructions=self.planner_instructions(tools_registry),
                ),
                model=base_model_planner,
                name="planner",
                workflow_id=self._workflow_id,
                http_client=self._http_client,
                system_prompt=PLANNER_PROMPT,
                toolset=planner_toolset,
                workflow_type=self._workflow_type,
            )

        return {
            "agent": planner,
            "toolset": planner_toolset,
            "supervisor": PlanSupervisorAgent(supervised_agent_name="planner"),
            "tools_executor": ToolsExecutor(
                tools_agent_name="planner",
                toolset=planner_toolset,
                workflow_id=self._workflow_id,
                workflow_type=self._workflow_type,
            ),
        }

    def _setup_workflow_graph(
        self,
        graph: StateGraph,
        executor_components,
        planner_components,
        tools_registry,
        goal,
    ):
        # Add nodes to the graph
        graph.set_entry_point("build_context")

        last_node_name = self._add_context_builder_nodes(graph, goal, tools_registry)
        disambiguation_component = GoalDisambiguationComponent(
            goal=goal,
            model=create_chat_model(
                max_tokens=MAX_TOKENS_TO_SAMPLE,
                config=self._model_config,
            ),
            http_client=self._http_client,
            workflow_id=self._workflow_id,
            tools_registry=tools_registry,
            allow_agent_to_request_user=self._workflow_config.get(
                "allow_agent_to_request_user", False
            ),
            workflow_type=self._workflow_type,
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
        graph.add_edge("update_plan", "planning")
        graph.add_edge("planning_supervisor", "planning")

        planner_approval_component = PlanApprovalComponent(
            workflow_id=self._workflow_id,
            approved_agent_name=planner_components["agent"].name,
            approved_agent_state=WorkflowStatusEnum.PLANNING,
        )

        planning_approval_entry_node = planner_approval_component.attach(
            graph=graph,
            next_node="set_status_to_execution",
            back_node="planning",
            exit_node="plan_terminator",
        )

        graph.add_conditional_edges(
            "planning",
            partial(_router, "planner", tools_registry),
            {
                Routes.CALL_TOOL: "update_plan",
                Routes.SUPERVISOR: "planning_supervisor",
                Routes.HANDOVER: planning_approval_entry_node,
                Routes.STOP: "plan_terminator",
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

        execution_approval_component = ToolsApprovalComponent(
            workflow_id=self._workflow_id,
            approved_agent_name=executor_components["agent"].name,
            approved_agent_state=WorkflowStatusEnum.EXECUTION,
            toolset=executor_components["toolset"],
        )

        execution_approval_entry_node = execution_approval_component.attach(
            graph=graph,
            next_node="execution_tools",
            back_node="execution",
            exit_node="plan_terminator",
        )

        graph.add_conditional_edges(
            "execution",
            partial(_router, "executor", tools_registry),
            {
                Routes.TOOLS_APPROVAL: execution_approval_entry_node,
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
        base_model_planner = create_chat_model(
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            config=self._model_config,
        )
        base_model_executor = create_chat_model(
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            config=self._model_config,
        )

        graph = StateGraph(WorkflowState)

        executor_components = self._setup_executor(
            goal, tools_registry, base_model_executor
        )
        planner_components = self._setup_planner(
            goal, tools_registry, base_model_planner, executor_components["toolset"]
        )

        graph = self._setup_workflow_graph(
            graph,
            executor_components,
            planner_components,
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
            context_elements=None,
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
        if "conversation_history" not in element:
            return

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

    def planner_instructions(self, tools_registry):
        available_feature_flags = current_feature_flag_context.get()
        if "batch_duo_workflow_planner_tasks" in available_feature_flags:
            self.log.info("Using batched planner")
            return PLANNER_TASK_BATCH_INSTRUCTIONS.format(
                create_plan_tool_name=tools_registry.get("create_plan").name,  # type: ignore
                add_new_task_tool_name=tools_registry.get("add_new_task").name,  # type: ignore
                remove_task_tool_name=tools_registry.get("remove_task").name,  # type: ignore
                update_task_description_tool_name=tools_registry.get("update_task_description").name,  # type: ignore
                get_plan_tool_name=tools_registry.get("get_plan").name,  # type: ignore
                handover_tool_name=HANDOVER_TOOL_NAME,
                project_id=self._project["id"],
                project_name=self._project["name"],
                project_url=self._project["http_url_to_repo"],
            )

        return PLANNER_INSTRUCTIONS.format(
            add_new_task_tool_name=tools_registry.get("add_new_task").name,  # type: ignore
            remove_task_tool_name=tools_registry.get("remove_task").name,  # type: ignore
            update_task_description_tool_name=tools_registry.get("update_task_description").name,  # type: ignore
            get_plan_tool_name=tools_registry.get("get_plan").name,  # type: ignore
            handover_tool_name=HANDOVER_TOOL_NAME,
            project_id=self._project["id"],
            project_name=self._project["name"],
            project_url=self._project["http_url_to_repo"],
        )

    def _setup_context_builder(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
    ):
        context_builder_toolset = tools_registry.toolset(CONTEXT_BUILDER_TOOLS)
        context_builder = Agent(
            goal=goal,
            model=create_chat_model(
                max_tokens=MAX_TOKENS_TO_SAMPLE,
                config=self._model_config,
            ),  # type: ignore
            name="context_builder",
            system_prompt=BUILD_CONTEXT_SYSTEM_MESSAGE.format(
                handover_tool_name=HANDOVER_TOOL_NAME,
                project_id=self._project["id"],
                project_name=self._project["name"],
                project_url=self._project["http_url_to_repo"],
            ),
            toolset=context_builder_toolset,
            workflow_id=self._workflow_id,
            http_client=self._http_client,
            workflow_type=self._workflow_type,
        )

        return {
            "agent": context_builder,
            "toolset": context_builder_toolset,
            "handover": HandoverAgent(
                new_status=WorkflowStatusEnum.PLANNING,
                handover_from=context_builder.name,
                include_conversation_history=True,
            ),
            "supervisor": PlanSupervisorAgent(supervised_agent_name="context_builder"),
            "tools_executor": ToolsExecutor(
                tools_agent_name=context_builder.name,
                toolset=context_builder_toolset,
                workflow_id=self._workflow_id,
                workflow_type=self._workflow_type,
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

        context_builder_approval_component = ToolsApprovalComponent(
            workflow_id=self._workflow_id,
            approved_agent_name="context_builder",
            approved_agent_state=WorkflowStatusEnum.NOT_STARTED,
            toolset=context_builder_components["toolset"],
        )

        context_builder_approval_entry_node = context_builder_approval_component.attach(
            graph=graph,
            next_node="build_context_tools",
            back_node="build_context",
            exit_node="plan_terminator",
        )

        graph.add_conditional_edges(
            "build_context",
            partial(_router, "context_builder", tools_registry),
            {
                Routes.CALL_TOOL: "build_context_tools",
                Routes.TOOLS_APPROVAL: context_builder_approval_entry_node,
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
