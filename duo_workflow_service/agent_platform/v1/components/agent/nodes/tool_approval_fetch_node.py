__all__ = ["ToolApprovalFetchNode"]

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.types import interrupt

from duo_workflow_service.agent_platform.v1.state import (
    FlowEvent,
    FlowEventType,
    FlowState,
    RuntimeIOKey,
)
from duo_workflow_service.entities import WorkflowStatusEnum


class ToolApprovalFetchNode:
    """Node that waits for approval decision and processes the response.

    This node:
    1. Interrupts workflow execution via interrupt()
    2. Waits for APPROVE, REJECT, or MODIFY event
    3. If approved: Continues to tool execution
    4. If rejected: Adds rejection ToolMessages to conversation history
    5. If modified: Adds rejection ToolMessages + user feedback HumanMessage.
        The user feedback UI chat log is emitted by the flow base via
        Command(update=...) to append alongside prior approval logs, matching
        the chat workflow pattern.
    6. Stores approval decision in output_key for router

    Args:
        name: Node name
        conversation_history_key: RuntimeIOKey for conversation history
        status_key: RuntimeIOKey for workflow status
        approval_decision_key: RuntimeIOKey for storing approval decision
    """

    def __init__(
        self,
        *,
        name: str,
        conversation_history_key: RuntimeIOKey,
        status_key: RuntimeIOKey,
        approval_decision_key: RuntimeIOKey,
    ):
        self.name = name
        self._conversation_history_key = conversation_history_key
        self._status_key = status_key
        self._approval_decision_key = approval_decision_key

    @staticmethod
    def _build_rejection_messages(tool_calls: list[ToolCall]) -> list[ToolMessage]:
        """Build rejection ToolMessages for each tool call.

        Args:
            tool_calls: List of ToolCall objects from AIMessage.tool_calls

        Returns:
            List of ToolMessages indicating rejection
        """
        return [
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=(
                    "Tool execution was rejected by user. "
                    "Do not proceed with this tool call. "
                    "Consider alternative approaches or ask the user for guidance."
                ),
            )
            for tool_call in tool_calls
        ]

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Wait for approval decision and process response."""

        # Interrupt workflow to wait for user input
        event: FlowEvent = interrupt("Workflow interrupted; waiting for tool approval.")

        # Get conversation history
        history_iokey = self._conversation_history_key.to_iokey(state)
        existing_history = history_iokey.value_from_state(state) or []

        approval_decision_iokey = self._approval_decision_key.to_iokey(state)

        if event["event_type"] == FlowEventType.APPROVE:
            # User approved - UI resolves the approval box, no UI log needed
            return {
                **self._status_key.to_nested_dict(WorkflowStatusEnum.EXECUTION, state),
                **approval_decision_iokey.to_nested_dict(FlowEventType.APPROVE),
            }

        if event["event_type"] == FlowEventType.REJECT:
            # User rejected - UI resolves the approval box, no UI log needed
            last_message = existing_history[-1]

            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                # Should not happen - request node validated this
                raise RuntimeError("No tool calls found to reject")

            rejection_messages = self._build_rejection_messages(last_message.tool_calls)

            history_dict = history_iokey.to_nested_dict(
                existing_history + rejection_messages
            )
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            decision_dict = approval_decision_iokey.to_nested_dict(FlowEventType.REJECT)
            return {**history_dict, **status_dict, **decision_dict}

        if event["event_type"] == FlowEventType.MODIFY:
            # User rejected with feedback. The user feedback UI chat log is
            # emitted by the flow base via Command(update=...) alongside the
            # resume event, matching the chat workflow pattern.
            last_message = existing_history[-1]

            if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
                # Should not happen - request node validated this
                raise RuntimeError("No tool calls found to reject")

            if "message" not in event or not event["message"]:
                raise ValueError(
                    "MODIFY event must include a message with user feedback"
                )

            rejection_messages = self._build_rejection_messages(last_message.tool_calls)
            user_feedback = HumanMessage(content=event["message"])

            history_dict = history_iokey.to_nested_dict(
                existing_history + rejection_messages + [user_feedback]
            )
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            decision_dict = approval_decision_iokey.to_nested_dict(FlowEventType.MODIFY)
            return {**history_dict, **status_dict, **decision_dict}

        # For any other event type, raise error
        raise ValueError(
            f"Unexpected event type for tool approval: {event['event_type']}. "
            f"Expected APPROVE, REJECT, or MODIFY."
        )
