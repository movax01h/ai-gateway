from datetime import datetime, timezone
from enum import auto
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service.agent_platform.v1.ui_log import (
    BaseUILogEvents,
    BaseUILogWriter,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)

__all__ = ["UILogEventsDeterministicStep", "UILogWriterDeterministicStep"]

from duo_workflow_service.tools import DuoBaseTool


class UILogEventsDeterministicStep(BaseUILogEvents):
    ON_TOOL_EXECUTION_SUCCESS = auto()
    ON_TOOL_EXECUTION_FAILED = auto()


class UILogWriterDeterministicStep(BaseUILogWriter[UILogEventsDeterministicStep]):

    @property
    def events_type(self) -> type[UILogEventsDeterministicStep]:
        return UILogEventsDeterministicStep

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
            content=self._format_message(tool, tool_call_args, tool_response),
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            tool_info=ToolInfo(name=tool.name, args=tool_call_args),
            message_sub_type=tool.name,
            correlation_id=correlation_id,
            additional_context=additional_context,
        )

    def _log_error(
        self,
        tool_name: str,
        error: str,
        correlation_id: Optional[str] = None,
        additional_context: Optional[list] = None,
        **kwargs,
    ) -> UiChatLog:
        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            message_sub_type=None,
            content=f"Tool {tool_name} execution failed: {error}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.FAILURE,
            correlation_id=correlation_id,
            tool_info=ToolInfo(name=tool_name, args={}),
            additional_context=additional_context,
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
                tool, tool_call_args, tool_response  # type: ignore[arg-type]
            )  # type: ignore[return-value]

        return tool.format_display_message(tool_call_args, tool_response)
