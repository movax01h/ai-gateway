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

__all__ = [
    "UILogEventsDeterministicStep",
    "UILogWriterDeterministicStep",
    "deterministic_step_ui_log_writer_class",
]

from duo_workflow_service.tools import DuoBaseTool


class UILogEventsDeterministicStep(BaseUILogEvents):
    ON_TOOL_EXECUTION_SUCCESS = auto()
    ON_TOOL_EXECUTION_FAILED = auto()


class UILogWriterDeterministicStep(BaseUILogWriter[UILogEventsDeterministicStep]):
    """A UI log writer for tool-execution events in deterministic step components.

    ``component_name`` is stored at construction time and embedded in every log
    entry, identifying the component that owns this writer.

    ``subsession_id`` is **not** stored at construction time — it must be supplied
    by the caller on each ``_log_success`` / ``_log_error`` invocation via
    ``**kwargs``.

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
    def events_type(self) -> type[UILogEventsDeterministicStep]:
        return UILogEventsDeterministicStep

    @override
    def _log_success(
        self,
        tool: BaseTool,
        tool_call_args: dict[str, Any],
        tool_response: Any = None,
        correlation_id: Optional[str] = None,
        additional_context: Optional[list] = None,
        **kwargs,
    ) -> UiChatLog:
        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            message_id=f"tool-{str(uuid4())}",
            content=self._format_message(tool, tool_call_args, tool_response),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            tool_info=build_tool_info(tool.name, tool_call_args, tool_response),
            message_sub_type=tool.name,
            correlation_id=correlation_id,
            additional_context=additional_context,
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
            message_id=f"tool-{str(uuid4())}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.FAILURE,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=ToolInfo(name=tool.name, args=tool_call_args),
            additional_context=kwargs.get("additional_context", []),
            message_sub_type=tool.name,
            component_name=self._component_name,
            subsession_id=kwargs.get("subsession_id"),
        )

    @staticmethod
    def _format_message(
        tool: BaseTool, tool_call_args: dict[str, Any], tool_response: Any = None
    ) -> str:
        if not hasattr(tool, "format_display_message"):
            args_str = ", ".join(f"{k}={str(v)}" for k, v in tool_call_args.items())
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


def deterministic_step_ui_log_writer_class(
    component_name: str,
) -> Callable[[UILogCallback], UILogWriterDeterministicStep]:
    """Factory that creates a ``UILogWriterDeterministicStep`` bound to *component_name*.

    The returned callable accepts a single ``UILogCallback`` argument, making it
    compatible with ``UIHistory.writer_class``.

    ``component_name`` is embedded in the writer and included in every log entry.
    ``subsession_id`` is not embedded — callers must pass it as a keyword argument to
    each ``log.success`` / ``log.error`` call.

    Args:
        component_name: Human-readable name of the component that owns this writer.
            Embedded in every log entry as ``component_name``.

    Returns:
        A partial that constructs a ``UILogWriterDeterministicStep`` when called with a
        ``UILogCallback``.
    """
    return partial(UILogWriterDeterministicStep, component_name=component_name)
