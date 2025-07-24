from enum import StrEnum
from typing import Any, Optional, Union

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from duo_workflow_service.agents import (
    Agent,
    HandoverAgent,
    PlanSupervisorAgent,
    ToolsExecutor,
)
from duo_workflow_service.components import PlanApprovalComponent, ToolsRegistry
from duo_workflow_service.components.planner.prompt import (
    HANDOVER_TOOL_NAME,
    PLANNER_GOAL,
    PLANNER_INSTRUCTIONS,
    PLANNER_PROMPT,
)
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.llm_factory import (
    AnthropicConfig,
    VertexConfig,
    create_chat_model,
)
from duo_workflow_service.workflows.abstract_workflow import MAX_TOKENS_TO_SAMPLE
from lib.internal_events.event_enum import CategoryEnum


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
        if last_message.tool_calls[0]["name"] == HANDOVER_TOOL_NAME:
            return Routes.HANDOVER
        return Routes.CALL_TOOL

    return Routes.SUPERVISOR


class PlannerComponent:
    def __init__(
        self,
        workflow_id: str,
        workflow_type: CategoryEnum,
        goal: str,
        planner_toolset: Any,
        executor_toolset: Any,
        tools_registry: ToolsRegistry,
        model_config: Union[AnthropicConfig, VertexConfig],
        project: Project,
        http_client: GitlabHttpClient,
    ):
        self.model_config = model_config
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.goal = goal
        self.planner_toolset = planner_toolset
        self.executor_toolset = executor_toolset
        self.tools_registry = tools_registry
        self.project = project
        self.http_client = http_client

    def attach(
        self,
        graph: StateGraph,
        exit_node: str,
        next_node: str,
        approval_component: Optional[PlanApprovalComponent],
    ):
        planner_toolset = self.planner_toolset
        base_model_planner = create_chat_model(
            max_tokens=MAX_TOKENS_TO_SAMPLE,
            config=self.model_config,
        )
        planner_agent_goal = PLANNER_GOAL.format(
            handover_tool_name=HANDOVER_TOOL_NAME,
            executor_agent_tools="\n".join(
                [
                    f"{tool_name}: {tool.description}"
                    for tool_name, tool in self.executor_toolset.items()
                ]
            ),
            goal=self.goal,
            create_plan_tool_name=self.tools_registry.get("create_plan").name,  # type: ignore
            get_plan_tool_name=self.tools_registry.get("get_plan").name,  # type: ignore
            add_new_task_tool_name=self.tools_registry.get("add_new_task").name,  # type: ignore
            remove_task_tool_name=self.tools_registry.get("remove_task").name,  # type: ignore
            update_task_description_tool_name=self.tools_registry.get("update_task_description").name,  # type: ignore
            planner_instructions=self.planner_instructions(self.tools_registry),
        )
        planner = Agent(
            goal=planner_agent_goal,
            model=base_model_planner,
            name="planner",
            workflow_id=self.workflow_id,
            http_client=self.http_client,
            system_prompt=PLANNER_PROMPT,
            toolset=planner_toolset,
            workflow_type=self.workflow_type,
        )
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

        graph.add_node("planning", planner.run)
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

    def planner_instructions(self, tools_registry):
        return PLANNER_INSTRUCTIONS.format(
            create_plan_tool_name=tools_registry.get("create_plan").name,  # type: ignore
            add_new_task_tool_name=tools_registry.get("add_new_task").name,  # type: ignore
            remove_task_tool_name=tools_registry.get("remove_task").name,  # type: ignore
            update_task_description_tool_name=tools_registry.get("update_task_description").name,  # type: ignore
            get_plan_tool_name=tools_registry.get("get_plan").name,  # type: ignore
            handover_tool_name=HANDOVER_TOOL_NAME,
            project_id=self.project["id"],
            project_name=self.project["name"],
            project_url=self.project["http_url_to_repo"],
        )
