from datetime import datetime, timezone
from typing import Annotated, Any, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools.handover import HandoverTool

__all__ = ["HandoverAgent"]


class HandoverAgent:
    _new_status: WorkflowStatusEnum
    _handover_from: str
    _include_conversation_history: bool

    def __init__(
        self,
        handover_from: Annotated[str, "Name of the agent that is handing over"],
        new_status: WorkflowStatusEnum,
        include_conversation_history: Annotated[
            bool,
            "Pass complete conversation history including summary from last message handover_tool call",
        ] = False,
    ):
        self._handover_from = handover_from
        self._new_status = new_status
        self._include_conversation_history = include_conversation_history

    async def run(self, state: DuoWorkflowStateType):
        handover_messages: List[BaseMessage] = []
        ui_chat_logs: List[UiChatLog] = []

        if (
            self._include_conversation_history
            and self._handover_from in state["conversation_history"]
        ):
            messages = self._replace_system_messages(
                state["conversation_history"][self._handover_from]
            )
            last_message = messages[-1]
            summary = self._extract_summary(last_message, ui_chat_logs)
            handover_messages = self._get_summary_to_handover(summary)

        result: dict[str, Any] = {
            "status": self._new_status,
            "ui_chat_log": ui_chat_logs,
        }

        # Only overwrite handover field if there are messages
        if handover_messages:
            result["handover"] = handover_messages
        return result

    def _replace_system_messages(
        self, messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        return [
            (
                HumanMessage(content=message.content)
                if isinstance(message, SystemMessage)
                else message
            )
            for message in messages
        ]

    def _extract_summary(
        self, last_message: BaseMessage, ui_chat_logs: List[UiChatLog]
    ) -> Optional[BaseMessage | None]:
        if not isinstance(last_message, AIMessage):
            return None

        handover_calls = [
            tool_call
            for tool_call in last_message.tool_calls
            if tool_call["name"] == HandoverTool.tool_title
        ]

        # make sure that there are no pending tool calls
        if len(handover_calls) > 0 and handover_calls[-1]["args"].get("summary"):
            summary = handover_calls[-1]["args"]["summary"]

            if summary:
                ui_chat_logs.append(
                    UiChatLog(
                        message_type=MessageTypeEnum.AGENT,
                        message_sub_type=None,
                        content=summary,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=None,
                        additional_context=None,
                        message_id=last_message.id,
                    )
                )
                return AIMessage(content=summary)

        return None

    def _get_summary_to_handover(
        self, summary: Optional[BaseMessage]
    ) -> List[BaseMessage]:
        if summary is not None and summary.content != "":
            return [summary]

        return []
