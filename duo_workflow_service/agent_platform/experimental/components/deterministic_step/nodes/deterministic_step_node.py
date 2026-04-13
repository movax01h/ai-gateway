from typing import Any

import structlog
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.experimental.components.deterministic_step.ui_log import (
    UILogEventsDeterministicStep,
    UILogWriterDeterministicStep,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    RuntimeIOKey,
    get_vars_from_state,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.exceptions import SecurityException
from duo_workflow_service.security.scanner_factory import apply_security_scanning
from lib.hidden_layer_log import set_hidden_layer_log_context
from lib.internal_events.event_enum import EventEnum

__all__ = ["DeterministicStepNode"]


TOOL_EXECUTION_STATUS_SUCCESS = "success"
TOOL_EXECUTION_STATUS_FAILED = "failed"


class DeterministicStepNode:
    def __init__(
        self,
        *,
        name: str,
        tool_name: str,
        inputs: list[IOKey | RuntimeIOKey],
        ui_history: UIHistory[
            UILogWriterDeterministicStep, UILogEventsDeterministicStep
        ],
        tool_responses_key: IOKey,
        tool_error_key: IOKey,
        execution_result_key: IOKey,
        validated_tool: BaseTool,
        tracker: ToolEventTracker,
    ):
        self.name = name
        self._tool_name = tool_name
        self._inputs = inputs
        self._logger = structlog.stdlib.get_logger("agent_platform")
        self._ui_history = ui_history
        self._tool_responses_key = tool_responses_key
        self._tool_error_key = tool_error_key
        self._execution_result_key = execution_result_key
        self._validated_tool = validated_tool
        self._tracker = tracker

    async def run(self, state: FlowState) -> dict:
        response, err_format, status = None, None, None
        tool_call_args = {}

        try:
            tool_call_args = get_vars_from_state(self._inputs, state)

            response = await self._execute_tool(
                tool=self._validated_tool, tool_call_args=tool_call_args
            )

            if not isinstance(response, (str, list, dict)):
                raise ValueError(
                    f"Invalid response type for tool {self._tool_name}: {response}"
                )

            status = TOOL_EXECUTION_STATUS_SUCCESS

        except Exception as e:
            status = TOOL_EXECUTION_STATUS_FAILED
            if isinstance(e, TypeError):
                err_format = self._tracker.handle_type_error_response(
                    tool=self._validated_tool, error=e
                )
            elif isinstance(e, ValidationError):
                err_format = self._tracker.handle_validation_error(
                    tool_name=self._tool_name, error=e
                )
            elif isinstance(e, SecurityException):
                err_format = e.format_user_message(self._tool_name)
            else:
                err_format = self._tracker.handle_execution_error(
                    tool_name=self._tool_name, error=e
                )

            response = getattr(e, "response", None)
            self._ui_history.log.error(
                tool=self._validated_tool,
                tool_call_args=tool_call_args,
                event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED,
                message=err_format if isinstance(e, SecurityException) else None,
                tool_response=f"{str(e)} {response}" if response else str(e),
            )

        result = {
            **self._ui_history.pop_state_updates(),
        }
        result = merge_nested_dict(
            result, self._tool_responses_key.to_nested_dict(response)
        )
        result = merge_nested_dict(
            result, self._tool_error_key.to_nested_dict(err_format)
        )
        result = merge_nested_dict(
            result, self._execution_result_key.to_nested_dict(status)
        )

        return result

    async def _execute_tool(
        self, tool_call_args: dict[str, Any], tool: BaseTool
    ) -> str | Any:
        with duo_workflow_metrics.time_tool_call(
            tool_name=tool.name, flow_type=self._tracker._flow_type.value
        ):
            tool_call_result = await tool.ainvoke(tool_call_args)

        set_hidden_layer_log_context(self._tool_name, tool_call_args)
        trust_level = getattr(tool, "trust_level", None)
        secure_result = apply_security_scanning(
            response=tool_call_result,
            tool_name=self._tool_name,
            trust_level=trust_level,
        )

        self._tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
            tool_name=tool.name,
        )

        self._ui_history.log.success(
            tool=tool,
            tool_call_args=tool_call_args,
            tool_response=secure_result,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS,
        )

        return secure_result
