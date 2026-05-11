from datetime import datetime, timezone
from functools import partial
from typing import Any, Callable, Literal, Optional, override

from duo_workflow_service.agent_platform.v1.ui_log.base import (
    BaseUILogEvents,
    BaseUILogWriter,
    UILogCallback,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus, UiChatLog

__all__ = [
    "DefaultUILogWriter",
    "default_ui_log_writer_class",
]


class DefaultUILogWriter[E: BaseUILogEvents](BaseUILogWriter[E]):
    """A UI log writer that emits log entries with a configurable ``message_type``.

    ``component_name`` is an optional parameter stored at construction time and
    embedded in every log entry, identifying the component that owns this writer.

    ``subsession_id`` is **not** stored at construction time — it must be supplied by
    the caller on each ``_log_success`` / ``_log_error`` invocation via
    ``**kwargs``.  This keeps the writer compatible with session-aware components
    and nodes (e.g. ``AgentComponent``) that resolve the subsession ID dynamically at
    runtime.

    When ``ui_role_as=MessageTypeEnum.TOOL`` is used, the writer also forwards
    ``tool_info``, ``message_sub_type``, and ``subsession_id`` from ``**kwargs`` into
    the log entry, making it suitable for delegation and subagent return nodes.

    Args:
        log_callback: Callback function that receives log entries.
        events_class: The ``BaseUILogEvents`` subclass that defines the valid events.
        ui_role_as: The ``MessageTypeEnum`` value to use as ``message_type`` in log
            entries (e.g. ``MessageTypeEnum.AGENT`` or ``MessageTypeEnum.TOOL``).
        component_name: Optional human-readable name of the component that owns this
            writer (e.g. ``"supervisor"``).  Embedded in every log entry as
            ``component_name`` when provided.
    """

    def __init__(
        self,
        log_callback: UILogCallback,
        events_class: type[E],
        ui_role_as: MessageTypeEnum,
        component_name: Optional[str] = None,
    ):
        super().__init__(log_callback)

        self._events_class = events_class
        self._ui_roles_as = ui_role_as
        self._component_name = component_name

    @property
    @override
    def events_type(self) -> type[E]:
        return self._events_class

    @override
    def _log_success(self, message: str, **kwargs: Any) -> UiChatLog:
        return self._build_log_entry(message, ToolStatus.SUCCESS, **kwargs)

    @override
    def _log_warning(self, *args: Any, **kwargs: Any) -> UiChatLog:
        raise NotImplementedError

    @override
    def _log_error(self, message: str, **kwargs: Any) -> UiChatLog:
        return self._build_log_entry(message, ToolStatus.FAILURE, **kwargs)

    def _build_log_entry(
        self, message: str, status: ToolStatus, **kwargs: Any
    ) -> UiChatLog:
        """Build a UI log entry with the configured message_type and component_name."""
        return UiChatLog(
            message_type=MessageTypeEnum(self._ui_roles_as),
            content=str(message),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=kwargs.get("tool_info"),
            additional_context=kwargs.get("additional_context", []),
            message_sub_type=kwargs.get("message_sub_type"),
            message_id=kwargs.get("message_id"),
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )


def default_ui_log_writer_class[E: BaseUILogEvents](
    events_class: type[E],
    ui_role_as: Literal["agent", "tool"],
    component_name: Optional[str] = None,
) -> Callable[[UILogCallback], DefaultUILogWriter[E]]:
    """Factory that creates a ``DefaultUILogWriter`` bound to *events_class*.

    The returned callable accepts a single ``UILogCallback`` argument, making it
    compatible with ``UIHistory.writer_class``.

    Args:
        events_class: The ``BaseUILogEvents`` subclass that defines the valid events.
        ui_role_as: The message type role (``"agent"`` or ``"tool"``).
        component_name: Optional human-readable name of the component that owns this
            writer.  Embedded in every log entry when provided.

    Returns:
        A partial that constructs a ``DefaultUILogWriter`` when called with a
        ``UILogCallback``.
    """
    return partial(
        DefaultUILogWriter,
        events_class=events_class,
        ui_role_as=MessageTypeEnum(ui_role_as),
        component_name=component_name,
    )
