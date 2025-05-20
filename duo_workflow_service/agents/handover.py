from datetime import datetime, timezone
from typing import Annotated, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from duo_workflow_service.agents.prompts import HANDOVER_TOOL_NAME
from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)

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
            handover_messages = [*messages[:-1], summary]

        if self._new_status == WorkflowStatusEnum.COMPLETED:
            ui_chat_logs.append(
                UiChatLog(
                    message_type=MessageTypeEnum.WORKFLOW_END,
                    content="Workflow completed successfully",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=None,
                    context_elements=None,
                )
            )

        return {
            "status": self._new_status,
            "handover": handover_messages,
            "ui_chat_log": ui_chat_logs,
        }

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
    ) -> Union[BaseMessage]:
        if not isinstance(last_message, AIMessage):
            return last_message

        handover_calls = [
            tool_call
            for tool_call in last_message.tool_calls
            if tool_call["name"] == HANDOVER_TOOL_NAME
        ]

        # make sure that there are no pending tool calls
        if len(handover_calls) > 0 and handover_calls[-1]["args"].get("summary"):
            summary = handover_calls[-1]["args"]["summary"]

            if summary:
                ui_chat_logs.append(
                    UiChatLog(
                        message_type=MessageTypeEnum.AGENT,
                        content=summary,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=None,
                        context_elements=None,
                    )
                )
                return HumanMessage(content=summary)

        return AIMessage(id=last_message.id, content=last_message.content)
