from enum import StrEnum
from functools import partial
from typing import Any, Optional, Union

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from duo_workflow_service.agents import (
    Agent,
    HandoverAgent,
    PlanSupervisorAgent,
    ToolsExecutor,
)
from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.executor.prompts import (
    EXECUTOR_SYSTEM_MESSAGE,
    GET_PLAN_TOOL_NAME,
    HANDOVER_TOOL_NAME,
    SET_TASK_STATUS_TOOL_NAME,
)
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.llm_factory import (
    AnthropicConfig,
    VertexConfig,
    create_chat_model,
)
from duo_workflow_service.workflows.abstract_workflow import MAX_TOKENS_TO_SAMPLE


class Routes(StrEnum):
    CALL_TOOL = "call_tool"
    TOOLS_APPROVAL = "tools_approval"
    SUPERVISOR = PlanSupervisorAgent.__name__
    HANDOVER = HandoverAgent.__name__
    STOP = "stop"


def _router(
    tool_registry: ToolsRegistry,
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    last_message = state["conversation_history"]["executor"][-1]
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


class ExecutorComponent:
    def __init__(
        self,
        workflow_id: str,
        workflow_type: CategoryEnum,
        goal: str,
        executor_toolset: Any,
        tools_registry: ToolsRegistry,
        model_config: Union[AnthropicConfig, VertexConfig],
        project: Any,
        http_client: GitlabHttpClient,
    ):
        self.model_config = model_config
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.goal = goal
        self.executor_toolset = executor_toolset
        self.tools_registry = tools_registry
        self.project = project
        self.http_client = http_client

    def attach(
        self,
        graph: StateGraph,
        exit_node: str,
        next_node: str,
        approval_component: Optional[ToolsApprovalComponent],
    ):
        base_model_executor = create_chat_model(
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            config=self.model_config,
        )
        agent = Agent(
            goal=self.goal,
            model=base_model_executor,
            name="executor",
            system_prompt=EXECUTOR_SYSTEM_MESSAGE.format(
                set_task_status_tool_name=SET_TASK_STATUS_TOOL_NAME,
                handover_tool_name=HANDOVER_TOOL_NAME,
                get_plan_tool_name=GET_PLAN_TOOL_NAME,
                project_id=self.project["id"],
                project_name=self.project["name"],
                project_url=self.project["http_url_to_repo"],
            ),
            toolset=self.executor_toolset,
            workflow_id=self.workflow_id,
            http_client=self.http_client,
            workflow_type=self.workflow_type,
        )
        tools_executor = ToolsExecutor(
            tools_agent_name="executor",
            toolset=self.executor_toolset,
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
        )
        handover = HandoverAgent(
            new_status=WorkflowStatusEnum.COMPLETED,
            handover_from="executor",
            include_conversation_history=True,
        )
        supervisor = PlanSupervisorAgent(supervised_agent_name="executor")
        # When tools approval component is not attached, proceed with tools execution
        tools_approval_entry_node = "execution_tools"
        if approval_component is not None:
            tools_approval_entry_node = approval_component.attach(
                graph=graph,
                next_node="execution_tools",
                back_node="execution",
                exit_node="plan_terminator",
            )

        graph.add_node("execution", agent.run)
        graph.add_node("execution_tools", tools_executor.run)
        graph.add_node("execution_supervisor", supervisor.run)
        graph.add_node("execution_handover", handover.run)

        graph.add_conditional_edges(
            "execution",
            partial(_router, self.tools_registry),
            {
                Routes.TOOLS_APPROVAL: tools_approval_entry_node,
                Routes.CALL_TOOL: "execution_tools",
                Routes.HANDOVER: "execution_handover",
                Routes.SUPERVISOR: "execution_supervisor",
                Routes.STOP: exit_node,
            },
        )
        graph.add_edge("execution_supervisor", "execution")
        graph.add_edge("execution_tools", "execution")
        graph.add_edge("execution_handover", next_node)

        return "execution"
