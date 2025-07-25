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
    FlowStateKeys,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import (
    PromptSecurity,
    SecurityException,
)
from duo_workflow_service.tools.toolset import Toolset
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventLabelEnum

__all__ = ["ToolNode"]


class ToolNode:
    def __init__(
        self,
        *,
        name: str,
        component_name: str,
        toolset: Toolset,
        flow_id: str,
        flow_type: CategoryEnum,
        internal_event_client: InternalEventsClient,
        ui_history: UIHistory[UILogWriterAgentTools, UILogEventsAgent],
    ):
        self.name = name
        self._component_name = component_name
        self._toolset = toolset
        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client
        self._logger = structlog.stdlib.get_logger("agent_platform")
        self._ui_history = ui_history

    async def run(self, state: FlowState) -> dict:
        conversation_history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self._component_name, []
        )

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

            tools_responses.append(
                ToolMessage(
                    content=self._sanitize_response(
                        response=response, tool_name=tool_name
                    ),
                    tool_call_id=tool_call_id,
                )
            )

        return {
            **self._ui_history.pop_state_updates(),
            FlowStateKeys.CONVERSATION_HISTORY: {
                self._component_name: tools_responses,
            },
        }

    async def _execute_tool(
        self, tool_call_args: dict[str, Any], tool: BaseTool
    ) -> str:
        try:
            with duo_workflow_metrics.time_tool_call(tool_name=tool.name):
                tool_call_result = await tool.arun(tool_call_args)

            self._track_internal_event(
                event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                tool_name=tool.name,
            )

            self._ui_history.log.success(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
            )

            return tool_call_result
        except Exception as e:
            self._ui_history.log.error(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            )

            if isinstance(e, TypeError):
                err_format = self._format_type_error_response(tool=tool, error=e)
            elif isinstance(e, ValidationError):
                err_format = self._format_validation_error(tool_name=tool.name, error=e)
            else:
                err_format = self._format_execution_error(tool_name=tool.name, error=e)

            return err_format

    def _sanitize_response(
        self, response: str | list | dict, tool_name: str
    ) -> str | list:
        try:
            sanitized = PromptSecurity.apply_security(
                response=response, tool_name=tool_name
            )
            if not isinstance(sanitized, (list, str)):
                raise ValueError("Sanitized response is neither string nor list")
            return sanitized
        except SecurityException as e:
            self._logger.error(f"Security validation failed for tool {tool_name}: {e}")
            raise

    def _track_internal_event(
        self,
        event_name: EventEnum,
        tool_name,
        extra=None,
    ):
        if extra is None:
            extra = {}
        additional_properties = InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
            property=tool_name,
            value=self._flow_id,
            **extra,
        )
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._flow_type,
        )

    def _format_type_error_response(self, tool: BaseTool, error: TypeError) -> str:
        if tool.args_schema:
            schema = f"The schema is: {tool.args_schema.model_json_schema()}"  # type: ignore[union-attr]
        else:
            schema = "The tool does not accept any argument"

        response = (
            f"Tool {tool.name} execution failed due to wrong arguments."
            f" You must adhere to the tool args schema! {schema}"
        )

        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool.name,
            extra={"error": str(error)},
        )

        return response

    def _format_validation_error(
        self,
        tool_name: str,
        error: ValidationError,
    ) -> str:
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={"error": str(error)},
        )
        return f"Tool {tool_name} raised validation error {str(error)}"

    def _format_execution_error(
        self,
        tool_name: str,
        error: Exception,
    ) -> str:
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={"error": str(error)},
        )

        return f"Tool runtime exception due to {str(error)}"
