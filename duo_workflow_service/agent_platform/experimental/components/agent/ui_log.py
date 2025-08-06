from datetime import datetime, timezone
from enum import auto
from typing import Any, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service.agent_platform.experimental.ui_log import (
    BaseUILogEvents,
    BaseUILogWriter,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
)
from duo_workflow_service.tools import DuoBaseTool

__all__ = [
    "UILogEventsAgent",
    "UILogWriterAgentTools",
]


class UILogEventsAgent(BaseUILogEvents):
    ON_AGENT_FINAL_ANSWER = auto()
    ON_TOOL_EXECUTION_SUCCESS = auto()
    ON_TOOL_EXECUTION_FAILED = auto()


class UILogWriterAgentTools(BaseUILogWriter):
    @property
    def events_type(self) -> type[UILogEventsAgent]:
        return UILogEventsAgent

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
            tool_info=ToolInfo(name=tool.name, args=tool_call_args),
            additional_context=kwargs.get("context_elements", []),
            message_sub_type=tool.name,
        )

    def _log_error(
        self,
        tool: BaseTool,
        tool_call_args: dict[str, Any],
        message: Optional[str] = None,
        **kwargs,
    ) -> UiChatLog:
        if not message:
            message = f"An error occurred when executing the tool: {
                self._format_message(tool, tool_call_args, kwargs.get('tool_response'))}"

        return UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.FAILURE,
            correlation_id=kwargs.get("correlation_id"),
            tool_info=ToolInfo(name=tool.name, args=tool_call_args),
            additional_context=kwargs.get("additional_context", []),
            message_sub_type=tool.name,
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
                # type: ignore[arg-type]
                parsed = schema(**tool_call_args)
                return tool.format_display_message(parsed, tool_response)
        except Exception:
            return DuoBaseTool.format_display_message(
                tool, tool_call_args, tool_response  # type: ignore[arg-type]
            )  # type: ignore[return-value]

        return tool.format_display_message(tool_call_args, tool_response)
