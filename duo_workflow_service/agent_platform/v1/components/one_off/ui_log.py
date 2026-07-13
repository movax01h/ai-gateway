from datetime import datetime, timezone
from enum import auto
from functools import partial
from typing import Any, Callable, Optional, override
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service.agent_platform.v1.ui_log import (
    BaseUILogEvents,
    BaseUILogWriter,
    UILogCallback,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
    build_tool_info,
)
from duo_workflow_service.tools import DuoBaseTool

__all__ = [
    "UILogEventsOneOff",
    "UILogWriterOneOffTools",
    "one_off_ui_log_writer_class",
]


class UILogEventsOneOff(BaseUILogEvents):
    ON_AGENT_FINAL_ANSWER = auto()
    ON_AGENT_REASONING = auto()
    ON_TOOL_CALL_INPUT = auto()
    ON_TOOL_EXECUTION_SUCCESS = auto()
    ON_TOOL_EXECUTION_FAILED = auto()


class UILogWriterOneOffTools(BaseUILogWriter):
    """A UI log writer for tool-execution events in one-off components.

    ``component_name`` is stored at construction time and embedded in every log
    entry, identifying the component that owns this writer.

    ``subsession_id`` is **not** stored at construction time — it must be supplied
    by the caller on each ``_log_success`` / ``_log_error`` invocation via
    ``**kwargs``.

    Callers may also pass ``message_id`` via ``**kwargs`` to ``_log_success`` /
    ``_log_error`` (e.g. the originating tool call's ``id``) so that a PENDING
    entry streamed earlier for the same tool call can be replaced in place by
    the final SUCCESS/FAILURE entry. When omitted, a random UUID is generated.

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
    def events_type(self) -> type[UILogEventsOneOff]:
        return UILogEventsOneOff

    @override
    def _log_success(
        self,
        tool: BaseTool,
        tool_call_args: dict[str, Any],
        message: Optional[str] = None,
        **kwargs,
    ) -> UiChatLog:
        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=message
            or self._format_message(tool, tool_call_args, kwargs.get("tool_response")),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=build_tool_info(
                tool.name, tool_call_args, kwargs.get("tool_response")
            ),
            additional_context=kwargs.get("context_elements", []),
            message_sub_type=tool.name,
            message_id=kwargs.get("message_id") or f"tool-{uuid4()!s}",
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )

    @override
    def _log_error(
        self,
        tool: BaseTool,
        tool_call_args: dict[str, Any],
        message: Optional[str] = None,
        **kwargs,
    ) -> UiChatLog:
        if not message:
            message = f"An error occurred when executing the tool: {
                self._format_message(tool, tool_call_args, kwargs.get('tool_response'))
            }"

        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.FAILURE,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=ToolInfo(name=tool.name, args=tool_call_args),
            additional_context=kwargs.get("context_elements", []),
            message_sub_type=tool.name,
            message_id=kwargs.get("message_id") or f"tool-{uuid4()!s}",
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )

    def _log_tool_call_input(
        self,
        tool: BaseTool,
        tool_call_args: dict[str, Any],
        message: Optional[str] = None,
        **kwargs,
    ) -> UiChatLog:
        """Log tool call arguments before execution."""
        if not message:
            args_str = ", ".join(f"{k}={v!s}" for k, v in tool_call_args.items())
            message = f"Calling tool '{tool.name}' with arguments: {args_str}"

        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.PENDING,  # Tool hasn't executed yet
            correlation_id=kwargs.get("correlation_id"),
            tool_info=ToolInfo(name=tool.name, args=tool_call_args),
            additional_context=kwargs.get("context_elements", []),
            message_sub_type=f"{tool.name}_input",
            message_id=f"tool-{uuid4()!s}",
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )

    def _log_warning(
        self,
        message: str,
        **kwargs,
    ) -> UiChatLog:
        """Log agent reasoning when no tools are called."""
        return UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            content=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=None,
            additional_context=kwargs.get("context_elements", []),
            message_sub_type="reasoning",
            message_id=f"agent-{uuid4()!s}",
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )

    @staticmethod
    def _format_message(
        tool: BaseTool, tool_call_args: dict[str, Any], tool_response: Any = None
    ) -> str:
        if not hasattr(tool, "format_display_message"):
            args_str = ", ".join(f"{k}={v!s}" for k, v in tool_call_args.items())
            return f"Using {tool.name}: {args_str}"

        try:
            schema = getattr(tool, "args_schema", None)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                parsed = schema(**tool_call_args)
                return tool.format_display_message(parsed, tool_response)
        except Exception:
            return DuoBaseTool.format_display_message(
                tool,  # type: ignore[arg-type]
                tool_call_args,
                tool_response,
            )  # type: ignore[return-value]

        return tool.format_display_message(tool_call_args, tool_response)


def one_off_ui_log_writer_class(
    component_name: str,
) -> Callable[[UILogCallback], UILogWriterOneOffTools]:
    """Factory that creates a ``UILogWriterOneOffTools`` bound to *component_name*.

    The returned callable accepts a single ``UILogCallback`` argument, making it
    compatible with ``UIHistory.writer_class``.

    ``component_name`` is embedded in the writer and included in every log entry.
    ``subsession_id`` is not embedded — callers must pass it as a keyword argument to
    each ``log.success`` / ``log.error`` call.

    Args:
        component_name: Human-readable name of the component that owns this writer.
            Embedded in every log entry as ``component_name``.

    Returns:
        A partial that constructs a ``UILogWriterOneOffTools`` when called with a
        ``UILogCallback``.
    """
    return partial(UILogWriterOneOffTools, component_name=component_name)
