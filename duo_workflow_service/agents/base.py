from datetime import datetime, timezone
from typing import Any

from ai_gateway.prompts import Input, Output, Prompt
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    SlashCommandStatus,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)
from lib.internal_events.event_enum import CategoryEnum


class BaseAgent(Prompt[Input, Output]):
    workflow_id: str
    workflow_type: CategoryEnum

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
            "agent_name": self.name,
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type.value,
        }
