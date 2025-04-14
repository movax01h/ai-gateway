import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt

from duo_workflow_service.entities import WorkflowEventType, WorkflowState
from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.internal_events.events_utils import track_workflow_event

log = structlog.get_logger("human_approval_check_executor")


class HumanApprovalCheckExecutor:
    _agent_name: str

    def __init__(self, agent_name: str, workflow_id: str) -> None:
        self._agent_name = agent_name
        self._workflow_id = workflow_id

    async def run(self, state: WorkflowState):
        # when using memory saver, human in the loop is not supported
        if not os.environ.get("WORKFLOW_INTERRUPT", False) or os.getenv("USE_MEMSAVER"):
            return {"status": state["status"]}

        ui_chat_logs: List[UiChatLog] = []
        event: WorkflowEvent = interrupt("Workflow interrupted")

        updates: Dict[str, Any] = {
            "last_human_input": event,
            "ui_chat_log": ui_chat_logs,
        }

        if event["event_type"] == WorkflowEventType.STOP:
            updates["status"] = WorkflowStatusEnum.CANCELLED

        # Track events based on event type
        track_workflow_event(
            event_type=event["event_type"],
            workflow_id=self._workflow_id,
            category=self.__class__.__name__,
            event_by_user=False,
        )

        if event["event_type"] == WorkflowEventType.MESSAGE:
            message = event["message"]
            correlation_id = (
                event["correlation_id"] if event.get("correlation_id") else None
            )

            ui_chat_logs.append(
                UiChatLog(
                    correlation_id=correlation_id,
                    message_type=MessageTypeEnum.USER,
                    content=f"Received message: {message}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    tool_info=None,
                )
            )

            # Check if last message was a tool call
            last_message = state["conversation_history"][self._agent_name][-1]
            messages: List[BaseMessage] = [
                ToolMessage(
                    content="Tool cancelled temporarily as user has a question",
                    tool_call_id=tool_call.get("id"),
                )
                for tool_call in getattr(last_message, "tool_calls", [])
            ]

            messages.append(HumanMessage(content=message))
            updates["conversation_history"] = {self._agent_name: messages}
        return updates
