import json
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from duo_workflow_service.agents import HandoverAgent, ToolsExecutor
from duo_workflow_service.agents.agent import build_agent
from duo_workflow_service.components.base import BaseComponent
from duo_workflow_service.entities import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.handover import HandoverTool


class Routes(StrEnum):
    END = "end"
    CALL_TOOL = "call_tool"
    HANDOVER = HandoverAgent.__name__
    CONTINUE = "continue"
    STOP = "stop"


AGENT_NAME = "create_branch"


def _router(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    last_message = state["conversation_history"][AGENT_NAME][-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        if last_message.tool_calls[0]["name"] == HandoverTool.tool_title:
            return Routes.HANDOVER
        return Routes.CALL_TOOL

    return Routes.STOP


def _should_continue(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.ERROR, WorkflowStatusEnum.CANCELLED]:
        return Routes.STOP

    return Routes.CONTINUE


class CreateRepositoryBranchComponent(BaseComponent):
    def __init__(
        self,
        toolset: Any,
        project: Project,
        **kwargs: Any,
    ):
        self.toolset = toolset
        self.project = project
        super().__init__(**kwargs)

    def attach(
        self,
        graph: StateGraph,
        exit_node: str,
        next_node: str,
    ):
        toolset = self.toolset
        branch_agent = build_agent(
            AGENT_NAME,
            self.prompt_registry,
            self.user,
            "workflow/create_branch",
            "^1.0.0",
            tools=toolset.bindable,
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
            http_client=self.http_client,
            prompt_template_inputs={
                "goal": self.goal,
                "ref": self._get_source_branch(),
                "workflow_id": self.workflow_id,
                "project_id": self.project["id"],
                "repository_url": self.project["http_url_to_repo"],
            },
        )

        graph.add_node("create_branch", branch_agent.run)

        tools_executor = ToolsExecutor(
            tools_agent_name=AGENT_NAME,
            toolset=toolset,
            workflow_id=self.workflow_id,
            workflow_type=self.workflow_type,
        )

        graph.add_node("create_branch_tools", tools_executor.run)
        graph.add_node(
            "create_branch_handover",
            HandoverAgent(
                new_status=WorkflowStatusEnum.NOT_STARTED,
                handover_from=AGENT_NAME,
                include_conversation_history=True,
            ).run,
        )

        graph.add_conditional_edges(
            "create_branch",
            _router,
            {
                Routes.CALL_TOOL: "create_branch_tools",
                Routes.HANDOVER: "create_branch_handover",
                Routes.STOP: exit_node,
            },
        )

        graph.add_conditional_edges(
            "create_branch_tools",
            _should_continue,
            {
                Routes.CONTINUE: "create_branch",
                Routes.STOP: exit_node,
            },
        )
        graph.add_edge("create_branch_handover", next_node)

        return "create_branch"  # entry node for create branch

    def _get_source_branch(self):
        if not self.additional_context:
            return self.project["default_branch"]

        for context in self.additional_context:
            if context.category == "agent_user_environment" and context.content:
                try:
                    content_data = json.loads(context.content)
                    return content_data.get("source_branch")
                except (json.JSONDecodeError, TypeError):
                    pass
        return self.project["default_branch"]
