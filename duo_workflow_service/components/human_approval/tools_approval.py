from typing import Literal

from langchain.tools import BaseTool
from langchain_core.messages import AIMessage

from duo_workflow_service.components.human_approval.component import (
    HumanApprovalComponent,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import format_tool_display_message


class ToolsApprovalComponent(HumanApprovalComponent):
    """Component for requesting human approval for tool executions."""

    _tools_registry: ToolsRegistry
    _approval_req_workflow_state: Literal[
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    ] = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
    _node_prefix: Literal["tools_approval"] = "tools_approval"

    def __init__(
        self,
        workflow_id: str,
        approved_agent_name: str,
        tools_registry: ToolsRegistry,
    ):
        super().__init__(
            workflow_id=workflow_id, approved_agent_name=approved_agent_name
        )
        self._tools_registry = tools_registry

    def _approval_message(self, state: WorkflowState) -> str:
        conversation = state["conversation_history"][self._approved_agent_name]
        last_message = conversation[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "Found no tool call requests to approve. If this situation persists, please file a bug report"

        tool_call_messages: list[str] = []
        for idx, call in enumerate(last_message.tool_calls):
            if not self._tools_registry.approval_required(call["name"]):
                continue

            if (
                tool := self._tools_registry.get(call["name"])
            ) is None or not isinstance(tool, BaseTool):
                continue

            if (msg := format_tool_display_message(tool, call["args"])) is None:
                continue

            tool_call_messages.append(f"{idx + 1}. {msg}")

        if len(tool_call_messages) == 0:
            raise RuntimeError("No valid tool calls were found to display.")

        tool_calls_msgs = "\n".join(tool_call_messages)

        return (
            "In order to complete the current task I would like to run following tools:\n\n"
            f"{tool_calls_msgs}\n\n"
            "In order to approve the execution, select Approve, "
            "select Deny to reject requested tool runs,"
            "otherwise provide your feedback via chat UI"
        )
