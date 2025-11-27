from datetime import datetime, timezone
from enum import auto
from typing import Optional

from duo_workflow_service.agent_platform.experimental.ui_log import (
    BaseUILogEvents,
    BaseUILogWriter,
)
from duo_workflow_service.entities import MessageTypeEnum, UiChatLog

__all__ = ["UILogEventsHumanInput", "AgentLogWriter", "UserLogWriter"]


class UILogEventsHumanInput(BaseUILogEvents):
    ON_USER_INPUT_PROMPT = auto()
    ON_USER_RESPONSE = auto()


class AgentLogWriter(BaseUILogWriter[UILogEventsHumanInput]):
    """UI log writer for agent messages in HumanInputComponent."""

    @property
    def events_type(self) -> type[UILogEventsHumanInput]:
        return UILogEventsHumanInput

    def _log_success(
        self,
        content: str,
        correlation_id: Optional[str] = None,
        additional_context: Optional[list] = None,
        **kwargs,
    ) -> UiChatLog:
        """Create a success UI log entry for agent messages."""
        return UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content=content,
            message_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=None,
            correlation_id=correlation_id,
            tool_info=None,
            additional_context=additional_context,
        )


class UserLogWriter(BaseUILogWriter[UILogEventsHumanInput]):
    """UI log writer for user messages in HumanInputComponent."""

    @property
    def events_type(self) -> type[UILogEventsHumanInput]:
        return UILogEventsHumanInput

    def _log_success(
        self,
        content: str,
        correlation_id: Optional[str] = None,
        additional_context: Optional[list] = None,
        **kwargs,
    ) -> UiChatLog:
        """Create a success UI log entry for user messages."""
        return UiChatLog(
            message_type=MessageTypeEnum.USER,
            message_sub_type=None,
            content=content,
            message_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=None,
            correlation_id=correlation_id,
            tool_info=None,
            additional_context=additional_context,
        )
