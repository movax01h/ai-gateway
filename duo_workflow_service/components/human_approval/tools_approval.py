from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage

from duo_workflow_service.components.human_approval.component import (
    HumanApprovalComponent,
)
from duo_workflow_service.entities.state import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import (
    MalformedToolCallError,
    Toolset,
    format_tool_display_message,
)
from lib import Result, result


class ToolsApprovalComponent(HumanApprovalComponent):
    """Component for requesting human approval for tool executions."""

    _toolset: Toolset
    _approval_req_workflow_state: Literal[
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    ] = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    _node_prefix: Literal["tools_approval"] = "tools_approval"

    def __init__(
        self,
        workflow_id: str,
        approved_agent_name: str,
        toolset: Toolset,
    ):
        super().__init__(
            workflow_id=workflow_id, approved_agent_name=approved_agent_name
        )
        self._toolset = toolset

    def _request_approval(self, state: WorkflowState):
        conversation = state["conversation_history"][self._approved_agent_name]
        last_message = conversation[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {
                "status": state["status"],
                "conversation_history": {
                    self._approved_agent_name: [
                        HumanMessage(
                            content="No tool calls has been found, please prepare tool calls for the current step in the plan"
                        )
                    ]
                },
            }

        valid_tool_calls, invalid_tool_calls = self._filter_valid_tool_calls(
            last_message.tool_calls
        )

        if len(invalid_tool_calls) > 0:
            return {
                "status": state["status"],
                "conversation_history": {
                    self._approved_agent_name: [
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            content="Tool call has been rejected due to other tool call in the last AIMessage being malformed",
                        )
                        for tool_call in valid_tool_calls
                    ]
                    + [
                        ToolMessage(
                            tool_call_id=tool_call_error.tool_call["id"],
                            content=str(tool_call_error),
                        )
                        for tool_call_error in invalid_tool_calls
                    ]
                },
            }
        return super()._request_approval(state)

    def _filter_valid_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> tuple[list[ToolCall], list[MalformedToolCallError]]:
        invalid_tool_calls = []
        valid_tool_calls = []

        for tool_call in tool_calls:
            try:
                self._toolset.validate_tool_call(tool_call)
                valid_tool_calls.append(tool_call)
            except MalformedToolCallError as e:
                # Add to invalid list if tool doesn't exist or has invalid arguments
                invalid_tool_calls.append(e)
        return valid_tool_calls, invalid_tool_calls

    def _build_approval_request(
        self, state: WorkflowState
    ) -> Result[str, RuntimeError]:
        conversation = state["conversation_history"][self._approved_agent_name]
        last_message = conversation[-1]

        tool_call_messages: list[str] = []
        for idx, call in enumerate(last_message.tool_calls):  # type: ignore[attr-defined]
            try:
                if self._toolset.approved(call["name"]):
                    continue

                tool = self._toolset[call["name"]]
                if (msg := format_tool_display_message(tool, call["args"])) is None:
                    continue

                tool_call_messages.append(f"{idx + 1}. {msg}")
            except KeyError:
                # tool call referred to NO-OP tool like HandOver tool which does not
                # require approvals
                continue

        if len(tool_call_messages) == 0:
            raise RuntimeError("No valid tool calls were found to display.")

        tool_calls_msgs = "\n".join(tool_call_messages)

        return result.Ok(
            "In order to complete the current task I would like to run following tools:\n\n"
            f"{tool_calls_msgs}\n\n"
            "In order to approve the execution, select Approve, "
            "select Deny to reject requested tool runs,"
            "otherwise provide your feedback via chat UI"
        )
