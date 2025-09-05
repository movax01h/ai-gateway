from enum import StrEnum
from functools import partial
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from ai_gateway.model_metadata import current_model_metadata_context
from duo_workflow_service.agents import (
    HandoverAgent,
    PlanSupervisorAgent,
    ToolsExecutor,
)
from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.base import BaseComponent
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.handover import HandoverTool


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
        if last_message.tool_calls[0]["name"] == HandoverTool.tool_title:
            return Routes.HANDOVER
        if any(
            tool_registry.approval_required(call["name"])
            for call in last_message.tool_calls
        ):
            return Routes.TOOLS_APPROVAL
        return Routes.CALL_TOOL

    return Routes.SUPERVISOR


class ExecutorComponent(BaseComponent):
    def __init__(self, executor_toolset: Any, project: Project, **kwargs: Any):
        self.executor_toolset = executor_toolset
        self.project = project
        super().__init__(**kwargs)

    def attach(
        self,
        graph: StateGraph,
        exit_node: str,
        next_node: str,
        approval_component: Optional[ToolsApprovalComponent],
    ):
        agent = self.prompt_registry.get_on_behalf(
            self.user,
            "workflow/executor",
            "^2.0.0",
            tools=self.executor_toolset.bindable,  # type: ignore[arg-type]
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
            http_client=self.http_client,
            model_metadata=current_model_metadata_context.get(),
            prompt_template_inputs={
                "set_task_status_tool_name": "set_task_status",
                "get_plan_tool_name": "get_plan",
            },
        )
        agent.prompt_template_inputs.setdefault("agent_user_environment", {}).update(
            self.agent_user_environment
        )
        graph.add_node("execution", agent.run)

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
