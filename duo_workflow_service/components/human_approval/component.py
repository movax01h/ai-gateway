# pylint: disable=direct-environment-variable-reference

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
from lib import Result, result

log = structlog.get_logger("human_approval_component")


class Routes(StrEnum):
    CONTINUE = "continue"
    BACK = "back"
    SKIP = "skip"
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
    def _build_approval_request(
        self, state: WorkflowState
    ) -> Result[str, RuntimeError]:
        """Prepares a request for approval, it returns a Result object indicating success or failure of the request
        preparation."""

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

        graph.add_conditional_edges(
            f"{self._node_prefix}_entry_{self._approved_agent_name}",
            self._approval_request_router,
            {
                Routes.CONTINUE: f"{self._node_prefix}_check_{self._approved_agent_name}",
                Routes.SKIP: next_node,
            },
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

    def _approval_request_router(
        self, state: WorkflowState
    ) -> Literal[Routes.SKIP, Routes.CONTINUE]:
        if state["status"] == self._approval_req_workflow_state:
            return Routes.CONTINUE
        return Routes.SKIP

    def _request_approval(self, state: WorkflowState):
        approval_request = self._build_approval_request(state)

        if not result.ok(approval_request):
            return {"status": state["status"]}

        ui_chat_logs = [
            UiChatLog(
                correlation_id=None,
                message_type=MessageTypeEnum.REQUEST,
                content=approval_request.value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                tool_info=None,
                context_elements=None,
            )
        ]

        return {
            "status": self._approval_req_workflow_state,
            "ui_chat_log": ui_chat_logs,
        }
