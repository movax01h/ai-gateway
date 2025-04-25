import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal

import structlog
from langgraph.graph import END, StateGraph

from duo_workflow_service.agents import HumanApprovalCheckExecutor
from duo_workflow_service.entities.event import WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)

log = structlog.get_logger("human_approval_component")


class Routes(StrEnum):
    CONTINUE = "continue"
    BACK = "back"
    STOP = "stop"


class HumanApprovalComponent(ABC):
    _node_prefix: str
    _approval_req_workflow_state: WorkflowStatusEnum

    def __init__(
        self,
        workflow_id: str,
        approved_agent_name: str,
    ):
        self._workflow_id = workflow_id
        self._approved_agent_name = approved_agent_name

    @abstractmethod
    def _approval_message(self, state: WorkflowState) -> str:
        """Returns the message content to display in the approval request UI.
        This message should explain what needs approval and how to proceed."""

    def attach(
        self,
        graph: StateGraph,
        exit_node: str,
        back_node: str,
        next_node: str = END,
    ) -> Annotated[str, "Entry node name"]:
        # Skip if human approval is disabled or using memory saver
        if (
            os.getenv("USE_MEMSAVER")
            or os.environ.get("WORKFLOW_INTERRUPT", "False").lower() != "true"
        ):
            return next_node

        graph.add_node(
            f"{self._node_prefix}_entry_{self._approved_agent_name}",
            self._request_approval,
        )
        graph.add_edge(
            f"{self._node_prefix}_entry_{self._approved_agent_name}",
            f"{self._node_prefix}_check_{self._approved_agent_name}",
        )

        graph.add_node(
            f"{self._node_prefix}_check_{self._approved_agent_name}",
            HumanApprovalCheckExecutor(
                agent_name=self._approved_agent_name, workflow_id=self._workflow_id
            ).run,
        )

        graph.add_conditional_edges(
            f"{self._node_prefix}_check_{self._approved_agent_name}",
            self._determine_next_step,
            {
                Routes.CONTINUE: next_node,
                Routes.BACK: back_node,
                Routes.STOP: exit_node,
            },
        )

        return f"{self._node_prefix}_entry_{self._approved_agent_name}"

    def _determine_next_step(
        self, state: WorkflowState
    ) -> Literal[Routes.CONTINUE, Routes.BACK, Routes.STOP]:
        if state.get("status") in [
            WorkflowStatusEnum.CANCELLED,
            WorkflowStatusEnum.ERROR,
        ]:
            return Routes.STOP

        if not (event := state.get("last_human_input", None)):
            return Routes.BACK

        if event.get("event_type") == WorkflowEventType.RESUME:
            return Routes.CONTINUE

        if event.get("event_type") == WorkflowEventType.STOP:
            return Routes.STOP

        return Routes.BACK

    def _request_approval(self, state):
        ui_chat_logs = [
            UiChatLog(
                correlation_id=None,
                message_type=MessageTypeEnum.REQUEST,
                content=self._approval_message(state),
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                tool_info=None,
            )
        ]

        return {
            "status": self._approval_req_workflow_state,
            "ui_chat_log": ui_chat_logs,
        }
