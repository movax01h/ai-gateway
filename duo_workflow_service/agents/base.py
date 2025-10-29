from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableBinding

from ai_gateway.prompts import Input, Output, Prompt
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    SlashCommandStatus,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)
from lib.internal_events.event_enum import CategoryEnum


class BaseAgent(RunnableBinding[Input, Output]):
    name: str
    prompt: Prompt
    workflow_id: str
    workflow_type: CategoryEnum

    def __init__(self, prompt: Prompt, **kwargs) -> None:
        super().__init__(
            prompt=prompt, bound=prompt, **kwargs
        )  # type: ignore[call-arg] # seems that mypy checks only against the immediate parent's init arguments

    def _create_ui_chat_log(
        self,
        content: str,
        message_type: MessageTypeEnum = MessageTypeEnum.AGENT,
        status: ToolStatus | SlashCommandStatus | None = None,
        tool_info: ToolInfo | None = None,
    ) -> UiChatLog:
        return UiChatLog(
            message_type=message_type,
            message_sub_type=None,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            correlation_id=None,
            tool_info=tool_info,
            additional_context=None,
        )

    @property
    def internal_event_extra(self) -> dict[str, Any]:
        return {
            "agent_name": self.prompt.name,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type.value,
        }
