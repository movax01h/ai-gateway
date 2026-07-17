__all__ = ["ToolApprovalRequestNode"]

from datetime import datetime, timezone
from typing import Any

import structlog.stdlib
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools import (
    MalformedToolCallError,
    Toolset,
    UnknownToolError,
    format_tool_display_message,
)

log = structlog.stdlib.get_logger(__name__)


class ToolApprovalRequestNode:
    """Node that validates tool calls and requests human approval.

    This node:
    1. Reads last AIMessage from conversation history
    2. Validates tool calls exist and are well-formed
    3. Filters pre-approved tools
    4. Generates UI chat log entries for approval
    5. Sets workflow status to INPUT_REQUIRED

    Args:
        name: Node name
        conversation_history_key: RuntimeIOKey for conversation history
        toolset: Toolset containing available tools
        pre_approved_tools: List of tool names that skip approval
        status_key: RuntimeIOKey for workflow status
        ui_history: UI logging history
    """

    def __init__(
        self,
        *,
        name: str,
        conversation_history_key: RuntimeIOKey,
        toolset: Toolset,
        pre_approved_tools: list[str],
        status_key: RuntimeIOKey,
        ui_history: UIHistory,
    ):
        self.name = name
        self._conversation_history_key = conversation_history_key
        self._toolset = toolset
        self._pre_approved_tools = set(pre_approved_tools)
        self._status_key = status_key
        self._ui_history = ui_history

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Validate tool calls and request approval."""
        # Get conversation history
        history_iokey = self._conversation_history_key.to_iokey(state)
        history = history_iokey.value_from_state(state) or []

        if not history:
            raise RuntimeError(
                f"No conversation history found for key {history_iokey.target}:{history_iokey.subkeys}"
            )

        last_message = history[-1]

        # Validate last message has tool calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            # Agent didn't generate tool calls - add error to history and return
            error_message = HumanMessage(
                content="No tool calls found. Please generate tool calls for the current task."
            )
            history_dict = history_iokey.to_nested_dict(history + [error_message])
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            return {**history_dict, **status_dict}

        # Filter and validate tool calls
        valid_calls, invalid_calls = self._filter_tool_calls(last_message.tool_calls)

        # If any tool calls are invalid, reject the entire batch.
        if invalid_calls:
            invalid_by_id = {e.tool_call["id"]: e for e in invalid_calls}
            error_messages = [
                ToolMessage(
                    tool_call_id=call["id"],
                    content=(
                        str(invalid_by_id[call["id"]])
                        if call["id"] in invalid_by_id
                        else "Tool call cancelled because another call in this batch was invalid."
                    ),
                )
                for call in last_message.tool_calls
            ]
            history_dict = history_iokey.to_nested_dict(history + error_messages)
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            return {**history_dict, **status_dict}

        # Filter out pre-approved tools
        needs_approval = [
            call for call in valid_calls if not self._should_skip_approval(call["name"])
        ]

        # If all tools are pre-approved, skip approval entirely
        if not needs_approval:
            # Set status to EXECUTION so router can explicitly route to tools
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            return status_dict

        # Build approval request UI messages
        ui_logs = self._build_approval_ui_logs(needs_approval)

        if not ui_logs:
            raise RuntimeError("No valid tool calls found to display for approval")

        result = self._status_key.to_nested_dict(
            WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED, state
        )

        return {**result, "ui_chat_log": ui_logs}

    def _filter_tool_calls(
        self, tool_calls: list
    ) -> tuple[list, list[MalformedToolCallError]]:
        """Filter tool calls into valid and invalid lists.

        Args:
            tool_calls: List of tool calls from AIMessage

        Returns:
            Tuple of (valid_calls, invalid_call_errors)
        """
        valid_calls = []
        invalid_calls = []

        for tool_call in tool_calls:
            try:
                self._toolset.validate_tool_call(tool_call)
                valid_calls.append(tool_call)
            except MalformedToolCallError as e:
                invalid_calls.append(e)

        return valid_calls, invalid_calls

    def _should_skip_approval(self, tool_name: str) -> bool:
        """Check if tool should skip approval.

        A tool skips approval if:
        1. It's in the component's pre_approved_tools list, OR
        2. It's in the toolset's pre-approved list
        """
        if tool_name in self._pre_approved_tools:
            return True

        try:
            return self._toolset.approved(tool_name)
        except UnknownToolError:
            # If tool doesn't exist, it will be caught by validation
            return False

    def _build_approval_ui_logs(self, tool_calls: list) -> list[UiChatLog]:
        """Build UI chat log entries for tool approval requests."""
        approval_messages = []

        for call in tool_calls:
            tool = self._toolset[call["name"]]

            # Get formatted display message for the tool
            msg = format_tool_display_message(tool, call["args"])
            if msg is None:
                continue

            ui_log = UiChatLog(
                correlation_id=None,
                message_type=MessageTypeEnum.REQUEST,
                message_sub_type=None,
                content=msg,
                message_id=call["id"],
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                tool_info=ToolInfo(name=call["name"], args=call["args"]),
                additional_context=None,
            )
            approval_messages.append(ui_log)

        return approval_messages
