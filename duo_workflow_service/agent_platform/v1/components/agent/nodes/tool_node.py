from typing import Any, Optional

import structlog
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.constants import NODE_ROLE_SEPARATOR
from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
)
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.agent_platform.v1.state.base import (
    BaseIOKey,
    NoneIOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import SecurityException
from duo_workflow_service.security.scanner_factory import apply_security_scanning
from duo_workflow_service.tools.toolset import Toolset
from lib.context import (
    is_orbit_tool,
    orbit_tool_call_count,
    total_tool_call_count,
)
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

    ``component_name`` is embedded in ``UiChatLog`` entries via the ``ui_history``
    writer (see ``agent_tools_ui_log_writer_class``).  The node itself does not
    store or forward ``component_name`` — it is the writer's responsibility.

    ``session_id_key`` is an ``IOKey`` (or ``NoneIOKey`` sentinel) that reads the
    active subsession ID from state.  When ``NoneIOKey`` (standalone mode),
    ``session_id`` is always ``None`` in log entries.  When an ``IOKey`` is
    provided (subagent mode), the resolved value is included in every log entry
    so the UI can attribute tool calls to the correct subsession.

    Args:
        name: LangGraph node name.
        conversation_history_key: ``RuntimeIOKey`` that resolves the
            conversation-history ``IOKey`` at runtime.
        toolset: Collection of tools available for execution.
        ui_history: UI log history writer for tool execution events.  Must be
            constructed with a writer that already has ``component_name`` bound
            (e.g. via ``agent_tools_ui_log_writer_class``).
        session_id_key: ``IOKey`` pointing to the active subsession ID in state.
            Defaults to ``NoneIOKey()`` for standalone components (always
            resolves to ``None``).
    """

    def __init__(
        self,
        *,
        name: str,
        toolset: Toolset,
        ui_history: UIHistory[UILogWriterAgentTools, UILogEventsAgent],
        conversation_history_key: RuntimeIOKey,
        tracker: ToolEventTracker,
        session_id_key: BaseIOKey = NoneIOKey(alias="session_id"),
    ):
        self.name = name
        self._toolset = toolset
        self._logger = structlog.stdlib.get_logger("agent_platform")
        self._ui_history = ui_history
        self._conversation_history_key = conversation_history_key
        self._tracker = tracker
        self._session_id_key = session_id_key

    def _resolve_session_id(self, state: FlowState) -> Optional[str]:
        """Resolve the active session ID from state.

        Returns:
            The session ID string when running as a subagent, or ``None`` for
            standalone components (when ``session_id_key`` is ``NoneIOKey``).
        """
        value = self._session_id_key.value_from_state(state)
        return str(value) if value is not None else None

    async def run(self, state: FlowState) -> dict:
        history_iokey = self._conversation_history_key.to_iokey(state)
        conversation_history = history_iokey.value_from_state(state) or []
        session_id = self._resolve_session_id(state)

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
                    tool=self._toolset[tool_name],
                    tool_call_args=tool_call_args,
                    session_id=session_id,
                )

            if not isinstance(response, (str, list, dict)):
                raise ValueError(
                    f"Invalid response type for tool {tool_name}: {response}"
                )

            tool = self._toolset.get(tool_name)
            set_hidden_layer_log_context(tool_name, tool_call_args)
            sanitized = self._sanitize_response(
                response=response, tool_name=tool_name, tool=tool, session_id=session_id
            )
            tools_responses.append(
                ToolMessage(
                    content=sanitized,  # type: ignore[arg-type]
                    tool_call_id=tool_call_id,
                )
            )

        # If any todo_write call was made, write its args to state so downstream
        # components can read the task list.
        # Component name is derived from the node name (format: "<component>#<role>").
        context_updates: dict = {}
        component_name = self.name.split(NODE_ROLE_SEPARATOR)[0]
        for tool_call in tool_calls:
            if tool_call["name"] == "todo_write":
                context_updates = {
                    "context": {
                        component_name: {"last_todo_write": tool_call.get("args", {})}
                    }
                }

        # Append tool responses to existing history for replace-based reducer.
        # The reducer will replace this component's conversation history with
        # the complete list returned here.
        return {
            **self._ui_history.pop_state_updates(),
            **history_iokey.to_nested_dict(conversation_history + tools_responses),
            **context_updates,
        }

    async def _execute_tool(
        self,
        tool_call_args: dict[str, Any],
        tool: BaseTool,
        session_id: Optional[str] = None,
    ) -> str:
        total_tool_call_count.set(total_tool_call_count.get() + 1)
        if is_orbit_tool(tool.name):
            orbit_tool_call_count.set(orbit_tool_call_count.get() + 1)

        try:
            with duo_workflow_metrics.time_tool_call(
                tool_name=tool.name, flow_type=self._tracker._flow_type.value
            ):
                tool_call_result = await tool.ainvoke(tool_call_args)

            self._tracker.track_internal_event(
                event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                tool_name=tool.name,
            )

            if is_orbit_tool(tool.name):
                self._tracker.track_internal_event(
                    event_name=EventEnum.ORBIT_DAP_TOOL_CALLED,
                    tool_name=tool.name,
                )

            self._ui_history.log.success(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
                tool_response=tool_call_result,
                subsession_id=session_id,
            )

            return tool_call_result
        except Exception as e:
            response = getattr(e, "response", None)
            self._ui_history.log.error(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
                tool_response=f"{e!s} {response}" if response else str(e),
                subsession_id=session_id,
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

            if is_orbit_tool(tool.name):
                self._tracker.track_internal_event(
                    event_name=EventEnum.ORBIT_DAP_TOOL_FAILED,
                    tool_name=tool.name,
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

            return err_format

    def _sanitize_response(
        self,
        response: str | dict | list,
        tool_name: str,
        tool: BaseTool | None = None,
        session_id: Optional[str] = None,
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
                    subsession_id=session_id,
                )
            return error_message
