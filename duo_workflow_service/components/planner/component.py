from enum import StrEnum
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from duo_workflow_service.agents import (
    HandoverAgent,
    PlanSupervisorAgent,
    ToolsExecutor,
)
from duo_workflow_service.agents.agent import build_agent
from duo_workflow_service.components.base import BaseComponent
from duo_workflow_service.components.human_approval.plan_approval import (
    PlanApprovalComponent,
)
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.entities.agent_user_environment import (
    process_agent_user_environment,
)
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.handover import HandoverTool


class Routes(StrEnum):
    END = "end"
    CALL_TOOL = "call_tool"
    TOOLS_APPROVAL = "tools_approval"
    SUPERVISOR = PlanSupervisorAgent.__name__
    HANDOVER = HandoverAgent.__name__
    STOP = "stop"


def _router(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    last_message = state["conversation_history"]["planner"][-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        if last_message.tool_calls[0]["name"] == HandoverTool.tool_title:
            return Routes.HANDOVER
        return Routes.CALL_TOOL

    return Routes.SUPERVISOR


class PlannerComponent(BaseComponent):
    def __init__(
        self,
        planner_toolset: Any,
        executor_toolset: Any,
        project: Project,
        **kwargs: Any,
    ):
        self.planner_toolset = planner_toolset
        self.executor_toolset = executor_toolset
        self.project = project
        super().__init__(**kwargs)

    def attach(
        self,
        graph: StateGraph,
        exit_node: str,
        next_node: str,
        approval_component: Optional[PlanApprovalComponent],
    ):
        planner_toolset = self.planner_toolset
        planner = build_agent(
            "planner",
            self.prompt_registry,
            self.user,
            "workflow/planner",
            "^1.0.0",
            tools=planner_toolset.bindable,
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
            http_client=self.http_client,
            prompt_template_inputs={
                "executor_agent_tools": "\n".join(
                    [
                        f"{tool_name}: {tool.description}"
                        for tool_name, tool in self.executor_toolset.items()
                    ]
                ),
                "create_plan_tool_name": self.tools_registry.get("create_plan").name,  # type: ignore
                "get_plan_tool_name": self.tools_registry.get("get_plan").name,  # type: ignore
                "add_new_task_tool_name": self.tools_registry.get("add_new_task").name,  # type: ignore
                "remove_task_tool_name": self.tools_registry.get("remove_task").name,  # type: ignore
                "update_task_description_tool_name": self.tools_registry.get(
                    "update_task_description"
                ).name,  # type: ignore
                "agent_user_environment": process_agent_user_environment(
                    self.additional_context
                ),
            },
        )

        graph.add_node("planning", planner.run)

        tools_executor = ToolsExecutor(
            tools_agent_name="planner",
            toolset=planner_toolset,
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
        )
        plan_supervisor = PlanSupervisorAgent(supervised_agent_name="planner")
        # When plan approval component is not attached, proceed to next node
        plan_approval_entry_node = next_node
        if approval_component is not None:
            plan_approval_entry_node = approval_component.attach(
                graph=graph,
                next_node="set_status_to_execution",
                back_node="planning",
                exit_node="plan_terminator",
            )

        graph.add_node("update_plan", tools_executor.run)
        graph.add_node("planning_supervisor", plan_supervisor.run)

        graph.add_conditional_edges(
            "planning",
            _router,
            {
                Routes.CALL_TOOL: "update_plan",
                Routes.SUPERVISOR: "planning_supervisor",
                Routes.HANDOVER: plan_approval_entry_node,
                Routes.STOP: exit_node,
            },
        )
        graph.add_edge("update_plan", "planning")
        graph.add_edge("planning_supervisor", "planning")

        return "planning"  # entry node for planner
