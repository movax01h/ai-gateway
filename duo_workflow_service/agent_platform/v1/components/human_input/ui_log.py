from datetime import datetime, timezone
from enum import auto
from functools import partial
from typing import Callable, Optional, override

from duo_workflow_service.agent_platform.v1.ui_log import (
    BaseUILogEvents,
    BaseUILogWriter,
    UILogCallback,
)
from duo_workflow_service.entities import MessageTypeEnum, UiChatLog

__all__ = [
    "UILogEventsHumanInput",
    "AgentLogWriter",
    "UserLogWriter",
    "agent_log_writer_class",
    "user_log_writer_class",
]


class UILogEventsHumanInput(BaseUILogEvents):
    ON_USER_INPUT_PROMPT = auto()
    ON_USER_RESPONSE = auto()


class AgentLogWriter(BaseUILogWriter[UILogEventsHumanInput]):
    """UI log writer for agent messages in HumanInputComponent.

    ``component_name`` is stored at construction time and embedded in every log
    entry, identifying the component that owns this writer.

    ``subsession_id`` is **not** stored at construction time — it must be supplied
    by the caller on each ``_log_success`` invocation via ``**kwargs``.

    Args:
        log_callback: Callback function that receives log entries.
        component_name: Human-readable name of the component that owns this writer.
            Embedded in every log entry as ``component_name``.
    """

    def __init__(
        self,
        log_callback: UILogCallback,
        component_name: str,
    ):
        super().__init__(log_callback)
        self._component_name = component_name

    @property
    @override
    def events_type(self) -> type[UILogEventsHumanInput]:
        return UILogEventsHumanInput

    @override
    def _log_success(
        self,
        content: str,
        request_type: str,
        correlation_id: Optional[str] = None,
        additional_context: Optional[list] = None,
        **kwargs,
    ) -> UiChatLog:
        """Create a success UI log entry for human input request messages."""
        return UiChatLog(
            message_type=MessageTypeEnum.REQUEST,
            message_sub_type=request_type,
            content=content,
            message_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=None,
            correlation_id=correlation_id,
            tool_info=None,
            additional_context=additional_context,
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )


class UserLogWriter(BaseUILogWriter[UILogEventsHumanInput]):
    """UI log writer for user messages in HumanInputComponent.

    ``component_name`` is stored at construction time and embedded in every log
    entry, identifying the component that owns this writer.

    ``subsession_id`` is **not** stored at construction time — it must be supplied
    by the caller on each ``_log_success`` invocation via ``**kwargs``.

    Args:
        log_callback: Callback function that receives log entries.
        component_name: Human-readable name of the component that owns this writer.
            Embedded in every log entry as ``component_name``.
    """

    def __init__(
        self,
        log_callback: UILogCallback,
        component_name: str,
    ):
        super().__init__(log_callback)
        self._component_name = component_name

    @property
    @override
    def events_type(self) -> type[UILogEventsHumanInput]:
        return UILogEventsHumanInput

    @override
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
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )


def agent_log_writer_class(
    component_name: str,
) -> Callable[[UILogCallback], AgentLogWriter]:
    """Factory that creates an ``AgentLogWriter`` bound to *component_name*.

    The returned callable accepts a single ``UILogCallback`` argument, making it
    compatible with ``UIHistory.writer_class``.

    ``component_name`` is embedded in the writer and included in every log entry.
    ``subsession_id`` is not embedded — callers must pass it as a keyword argument to
    each ``log.success`` call.

    Args:
        component_name: Human-readable name of the component that owns this writer.
            Embedded in every log entry as ``component_name``.

    Returns:
        A partial that constructs an ``AgentLogWriter`` when called with a
        ``UILogCallback``.
    """
    return partial(AgentLogWriter, component_name=component_name)


def user_log_writer_class(
    component_name: str,
) -> Callable[[UILogCallback], UserLogWriter]:
    """Factory that creates a ``UserLogWriter`` bound to *component_name*.

    The returned callable accepts a single ``UILogCallback`` argument, making it
    compatible with ``UIHistory.writer_class``.

    ``component_name`` is embedded in the writer and included in every log entry.
    ``subsession_id`` is not embedded — callers must pass it as a keyword argument to
    each ``log.success`` call.

    Args:
        component_name: Human-readable name of the component that owns this writer.
            Embedded in every log entry as ``component_name``.

    Returns:
        A partial that constructs a ``UserLogWriter`` when called with a
        ``UILogCallback``.
    """
    return partial(UserLogWriter, component_name=component_name)
