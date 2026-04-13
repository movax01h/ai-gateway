from typing import Any

import structlog
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.experimental.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
)
from duo_workflow_service.agent_platform.experimental.state.base import RuntimeIOKey
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import SecurityException
from duo_workflow_service.security.scanner_factory import apply_security_scanning
from duo_workflow_service.tools.toolset import Toolset
from lib.hidden_layer_log import set_hidden_layer_log_context
from lib.internal_events.event_enum import EventEnum

__all__ = ["ToolNode"]


class ToolNode:
    """LangGraph node that executes tool calls from the last AIMessage in conversation history.

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary
    access.

    The conversation-history ``IOKey`` is resolved dynamically at runtime via
    ``conversation_history_key``.  This supports both the common case (static key
    wrapped in a ``RuntimeIOKey`` by the caller) and the supervisor case where
    the key is only known at runtime.

    Args:
        name: LangGraph node name.
        conversation_history_key: ``RuntimeIOKey`` that resolves the
            conversation-history ``IOKey`` at runtime.
        toolset: Collection of tools available for execution.
        ui_history: UI log history writer for tool execution events.
    """

    def __init__(
        self,
        *,
        name: str,
        toolset: Toolset,
        ui_history: UIHistory[UILogWriterAgentTools, UILogEventsAgent],
        conversation_history_key: RuntimeIOKey,
        tracker: ToolEventTracker,
    ):
        self.name = name
        self._toolset = toolset
        self._logger = structlog.stdlib.get_logger("agent_platform")
        self._ui_history = ui_history
        self._conversation_history_key = conversation_history_key
        self._tracker = tracker

    async def run(self, state: FlowState) -> dict:
        history_iokey = self._conversation_history_key.to_iokey(state)
        conversation_history = history_iokey.value_from_state(state) or []

        # TODO: add ability to register all tool calls in a follow up
        # context = state["context"].get(self.component_name, {})
        # context.setdefault("tool_calls", [])

        last_message = conversation_history[-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        tools_responses = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            if tool_name not in self._toolset:
                response = f"Tool {tool_name} not found"
            else:
                response = await self._execute_tool(
                    tool=self._toolset[tool_name], tool_call_args=tool_call_args
                )

            if not isinstance(response, (str, list, dict)):
                raise ValueError(
                    f"Invalid response type for tool {tool_name}: {response}"
                )

            tool = self._toolset.get(tool_name)
            set_hidden_layer_log_context(tool_name, tool_call_args)
            sanitized = self._sanitize_response(
                response=response, tool_name=tool_name, tool=tool
            )
            tools_responses.append(
                ToolMessage(
                    content=sanitized,  # type: ignore[arg-type]
                    tool_call_id=tool_call_id,
                )
            )

        # Append tool responses to existing history for replace-based reducer.
        # The reducer will replace this component's conversation history with
        # the complete list returned here.
        return {
            **self._ui_history.pop_state_updates(),
            **history_iokey.to_nested_dict(conversation_history + tools_responses),
        }

    async def _execute_tool(
        self, tool_call_args: dict[str, Any], tool: BaseTool
    ) -> str:
        try:
            with duo_workflow_metrics.time_tool_call(
                tool_name=tool.name, flow_type=self._tracker._flow_type.value
            ):
                tool_call_result = await tool.ainvoke(tool_call_args)

            self._tracker.track_internal_event(
                event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                tool_name=tool.name,
            )

            self._ui_history.log.success(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
                tool_response=tool_call_result,
            )

            return tool_call_result
        except Exception as e:
            response = getattr(e, "response", None)
            self._ui_history.log.error(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
                tool_response=f"{str(e)} {response}" if response else str(e),
            )

            if isinstance(e, TypeError):
                err_format = self._tracker.handle_type_error_response(
                    tool=tool, error=e
                )
            elif isinstance(e, ValidationError):
                err_format = self._tracker.handle_validation_error(
                    tool_name=tool.name, error=e
                )
            else:
                err_format = self._tracker.handle_execution_error(
                    tool_name=tool.name, error=e
                )

            return err_format

    def _sanitize_response(
        self,
        response: str | dict | list,
        tool_name: str,
        tool: BaseTool | None = None,
    ) -> str | dict | list:
        try:
            trust_level = getattr(tool, "trust_level", None)
            return apply_security_scanning(
                response=response,
                tool_name=tool_name,
                trust_level=trust_level,
            )
        except SecurityException as e:
            self._logger.error(f"Security validation failed for tool {tool_name}: {e}")
            error_message = e.format_user_message(tool_name)
            if tool is not None:
                self._ui_history.log.error(
                    tool=tool,
                    tool_call_args={},
                    message=error_message,
                    event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
                )
            return error_message
