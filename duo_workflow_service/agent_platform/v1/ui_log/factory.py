from datetime import datetime, timezone
from functools import partial
from typing import Callable, Literal

from duo_workflow_service.agent_platform.v1.ui_log.base import (
    BaseUILogEvents,
    BaseUILogWriter,
    UILogCallback,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus, UiChatLog

__all__ = ["DefaultUILogWriter", "default_ui_log_writer_class"]


class DefaultUILogWriter[E: BaseUILogEvents](BaseUILogWriter[E]):
    def __init__(
        self,
        log_callback: UILogCallback,
        events_class: type[E],
        ui_role_as: MessageTypeEnum,
    ):
        super().__init__(log_callback)

        self._events_class = events_class
        self._ui_roles_as = ui_role_as

    @property
    def events_type(self) -> type[E]:
        return self._events_class

    def _log_success(self, message: str, **kwargs) -> UiChatLog:
        return UiChatLog(
            message_type=MessageTypeEnum(self._ui_roles_as),
            content=str(message),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=None,
            additional_context=kwargs.get("additional_context", []),
            message_sub_type=None,
        )

    def _log_warning(self, *args, **kwargs) -> UiChatLog:
        raise NotImplementedError

    def _log_error(self, *args, **kwargs) -> UiChatLog:
        raise NotImplementedError


def default_ui_log_writer_class[E: BaseUILogEvents](
    events_class: type[E], ui_role_as: Literal["agent", "tool"]
) -> Callable[[UILogCallback], DefaultUILogWriter[E]]:
    return partial(
        DefaultUILogWriter,
        events_class=events_class,
        ui_role_as=MessageTypeEnum(ui_role_as),
    )
