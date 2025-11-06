from typing import Any

import structlog
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.v1.components.deterministic_step.ui_log import (
    UILogEventsDeterministicStep,
    UILogWriterDeterministicStep,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    get_vars_from_state,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import PromptSecurity
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventLabelEnum

__all__ = ["DeterministicStepNode"]


TOOL_EXECUTION_STATUS_SUCCESS = "success"
TOOL_EXECUTION_STATUS_FAILED = "failed"


# pylint: disable-next=too-many-instance-attributes
class DeterministicStepNode:
    def __init__(
        self,
        *,
        name: str,
        tool_name: str,
        inputs: list[IOKey],
        flow_id: str,
        flow_type: CategoryEnum,
        internal_event_client: InternalEventsClient,
        ui_history: UIHistory[
            UILogWriterDeterministicStep, UILogEventsDeterministicStep
        ],
        tool_responses_key: IOKey,
        tool_error_key: IOKey,
        execution_result_key: IOKey,
        validated_tool: BaseTool,
    ):
        self.name = name
        self._tool_name = tool_name
        self._inputs = inputs
        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client
        self._logger = structlog.stdlib.get_logger("agent_platform")
        self._ui_history = ui_history
        self._tool_responses_key = tool_responses_key
        self._tool_error_key = tool_error_key
        self._execution_result_key = execution_result_key
        self._validated_tool = validated_tool

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
                err_format = self._format_type_error_response(
                    tool=self._validated_tool, error=e
                )
            elif isinstance(e, ValidationError):
                err_format = self._format_validation_error(
                    tool_name=self._tool_name, error=e
                )
            else:
                err_format = self._format_execution_error(
                    tool_name=self._tool_name, error=e
                )

            response = getattr(e, "response", None)
            self._ui_history.log.error(
                tool=self._validated_tool,
                tool_call_args=tool_call_args,
                event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED,
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
            tool_name=tool.name, flow_type=self._flow_type.value
        ):
            tool_call_result = await tool.arun(tool_call_args)

        secure_result = PromptSecurity.apply_security_to_tool_response(
            response=tool_call_result, tool_name=self._tool_name
        )

        self._track_internal_event(
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
        self._record_metric(
            event_name=event_name,
            additional_properties=additional_properties,
        )
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._flow_type.value,
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
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
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
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
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
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        return f"Tool runtime exception due to {str(error)}"

    def _record_metric(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ) -> None:

        if event_name == EventEnum.WORKFLOW_TOOL_FAILURE:
            tool_name = additional_properties.property or "unknown"
            failure_reason = additional_properties.extra.get("error_type", "unknown")
            duo_workflow_metrics.count_agent_platform_tool_failure(
                flow_type=self._flow_type.value,
                tool_name=tool_name,
                failure_reason=failure_reason,
            )
