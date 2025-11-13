from datetime import datetime, timezone

from langchain_core.runnables import RunnableBinding

from ai_gateway.prompts import Input, Output, Prompt
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    SlashCommandStatus,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)


class BaseAgent(RunnableBinding[Input, Output]):
    name: str
    prompt: Prompt
    workflow_id: str

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
