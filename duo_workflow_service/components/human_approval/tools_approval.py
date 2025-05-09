from typing import Literal

from langchain_core.messages import AIMessage

from duo_workflow_service.components.human_approval.component import (
    HumanApprovalComponent,
)
from duo_workflow_service.entities.state import WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools import (
    Toolset,
    UnknownToolError,
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

    def _build_approval_request(
        self, state: WorkflowState
    ) -> Result[str, RuntimeError]:
        conversation = state["conversation_history"][self._approved_agent_name]
        last_message = conversation[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return result.Error(
                RuntimeError(
                    "Found no tool call requests to approve. If this situation persists, please file a bug report"
                )
            )

        tool_call_messages: list[str] = []
        unknown_tools = False
        for idx, call in enumerate(last_message.tool_calls):
            try:
                if self._toolset.approved(call["name"]):
                    continue

                tool = self._toolset[call["name"]]

                if (msg := format_tool_display_message(tool, call["args"])) is None:
                    continue

                tool_call_messages.append(f"{idx + 1}. {msg}")
            except UnknownToolError:
                unknown_tools = True
                continue
            except KeyError:
                # tool call refered to NO-OP tool like HandOver tool which does not
                # require approvals
                continue

        if not unknown_tools and len(tool_call_messages) == 0:
            raise RuntimeError("No valid tool calls were found to display.")

        # ignore unknown tools and let ToolsExecutor handle them
        if unknown_tools and len(tool_call_messages) == 0:
            return result.Error(
                RuntimeError(
                    "Unknown tool calls were found. No other valid tool calls were found to display."
                )
            )

        tool_calls_msgs = "\n".join(tool_call_messages)

        return result.Ok(
            "In order to complete the current task I would like to run following tools:\n\n"
            f"{tool_calls_msgs}\n\n"
            "In order to approve the execution, select Approve, "
            "select Deny to reject requested tool runs,"
            "otherwise provide your feedback via chat UI"
        )
