from enum import StrEnum
from functools import partial
from typing import Any, Optional, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from ai_gateway.model_metadata import current_model_metadata_context
from duo_workflow_service.agents import (
    Agent,
    AgentV2,
    HandoverAgent,
    PlanSupervisorAgent,
    ToolsExecutor,
)
from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.base import BaseComponent
from duo_workflow_service.components.executor.prompts import (
    DEPRECATED_OS_INFORMATION_COMPONENT,
    EXECUTOR_SYSTEM_MESSAGE,
    GET_PLAN_TOOL_NAME,
    HANDOVER_TOOL_NAME,
    OS_INFORMATION_COMPONENT,
    SET_TASK_STATUS_TOOL_NAME,
)
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.llm_factory import create_chat_model
from duo_workflow_service.workflows.abstract_workflow import MAX_TOKENS_TO_SAMPLE
from duo_workflow_service.workflows.type_definitions import OsInformationContext
from lib.feature_flags.context import FeatureFlag, is_feature_enabled


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
        if is_feature_enabled(FeatureFlag.DUO_WORKFLOW_PROMPT_REGISTRY):
            agent_v2: AgentV2 = cast(
                AgentV2,
                self.prompt_registry.get_on_behalf(
                    self.user,
                    "workflow/executor",
                    "^2.0.0",
                    tools=self.executor_toolset.bindable,  # type: ignore[arg-type]
                    workflow_id=self.workflow_id,
                    workflow_type=self.workflow_type,
                    http_client=self.http_client,
                    model_metadata=current_model_metadata_context.get(),
                ),
            )
            agent_v2.prompt_template_inputs.setdefault(
                "agent_user_environment", {}
            ).update(self.agent_user_environment)
            graph.add_node("execution", agent_v2.run)
        else:
            base_model_executor = create_chat_model(
                max_tokens=MAX_TOKENS_TO_SAMPLE,
                config=self.model_config,
            )
            agent = Agent(
                goal=self.goal,
                model=base_model_executor,
                name="executor",
                system_prompt=self._format_system_prompt(),
                toolset=self.executor_toolset,
                workflow_id=self.workflow_id,
                http_client=self.http_client,
                workflow_type=self.workflow_type,
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

    def _format_system_prompt(self) -> str:
        os_information = ""
        for context_type, context in self.agent_user_environment.items():
            if context_type == "os_information_context" and isinstance(
                context, OsInformationContext
            ):
                os_information = OS_INFORMATION_COMPONENT.format(
                    platform=context.platform,
                    architecture=context.architecture,
                )
        # Temporary support for deprecated os_information message
        if not os_information:
            for additional_context in self.additional_context or []:
                # We only want to add os_information if it's not empty
                if (
                    additional_context.category == "os_information"
                    and additional_context.content
                ):
                    os_information = DEPRECATED_OS_INFORMATION_COMPONENT.format(
                        os_information=additional_context.content
                    )

        return EXECUTOR_SYSTEM_MESSAGE.format(
            set_task_status_tool_name=SET_TASK_STATUS_TOOL_NAME,
            handover_tool_name=HANDOVER_TOOL_NAME,
            get_plan_tool_name=GET_PLAN_TOOL_NAME,
            project_id=self.project["id"],
            project_name=self.project["name"],
            project_url=self.project["http_url_to_repo"],
            os_information=os_information,
        )
