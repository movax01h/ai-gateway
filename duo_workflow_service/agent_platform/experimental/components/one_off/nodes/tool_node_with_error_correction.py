import re
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.experimental.components.one_off.ui_log import (
    UILogEventsOneOff,
    UILogWriterOneOffTools,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.experimental.ui_log import UIHistory
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import SecurityException
from duo_workflow_service.security.scanner_factory import apply_security_scanning
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum, EventLabelEnum


class ToolNodeWithErrorCorrection:
    """Enhanced ToolNode that tracks errors and provides feedback for correction loops."""

    def __init__(
        self,
        *,
        name: str,
        component_name: str,
        toolset: Toolset,
        flow_id: str,
        flow_type: GLReportingEventContext,
        internal_event_client: InternalEventsClient,
        ui_history: UIHistory[UILogWriterOneOffTools, UILogEventsOneOff],
        max_correction_attempts: int = 3,
        tool_calls_key: IOKey | None = None,
        tool_responses_key: IOKey | None = None,
        execution_result_key: IOKey | None = None,
    ):
        self.name = name
        self._component_name = component_name
        self._toolset = toolset
        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client
        self._logger = structlog.stdlib.get_logger("agent_platform")
        self._ui_history = ui_history
        self.max_correction_attempts = max_correction_attempts
        self.tool_calls_key = tool_calls_key
        self.tool_responses_key = tool_responses_key
        self.execution_result_key = execution_result_key

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute tools with error correction tracking."""
        # Get current context for error tracking
        context = state.get("context", {}).get(self._component_name, {})
        attempts = context.get("correction_attempts", 0)

        # Get conversation history
        conversation_history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self._component_name, []
        )

        # Get tool calls from the last message
        last_message = conversation_history[-1] if conversation_history else None
        tool_calls = getattr(last_message, "tool_calls", []) if last_message else []
        tool_responses = []

        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            if tool_name not in self._toolset:
                response = f"Tool {tool_name} not found"
            else:
                self._ui_history.log._log_tool_call_input(
                    tool=self._toolset[tool_name],
                    tool_call_args=tool_call_args,
                    event=UILogEventsOneOff.ON_TOOL_CALL_INPUT,
                )

                response = await self._execute_tool(
                    tool=self._toolset[tool_name],
                    tool_call_args=tool_call_args,
                )

            tool = self._toolset.get(tool_name)
            sanitized = self._sanitize_response(
                response=response, tool_name=tool_name, tool=tool
            )
            tool_responses.append(
                ToolMessage(
                    content=sanitized,  # type: ignore[arg-type]
                    tool_call_id=tool_call_id,
                )
            )

        result = {
            **self._ui_history.pop_state_updates(),
            FlowStateKeys.CONVERSATION_HISTORY: {
                self._component_name: tool_responses,
            },
        }

        # Store tool calls and responses using IOKeys if provided
        # Use merge_nested_dict to properly handle multiple IOKeys (like get_vars_from_state does)
        if self.tool_calls_key and tool_calls:
            tool_calls_dict = self.tool_calls_key.to_nested_dict(tool_calls)
            result = merge_nested_dict(result, tool_calls_dict)
        if self.tool_responses_key and tool_responses:
            tool_responses_dict = self.tool_responses_key.to_nested_dict(tool_responses)
            result = merge_nested_dict(result, tool_responses_dict)

        # Check for errors in tool responses
        errors = self._extract_errors_from_responses(tool_responses)

        if errors:
            # Create error feedback message for LLM
            error_feedback = self._create_error_feedback(
                errors, tool_calls, attempts + 1
            )

            # Update conversation_history while preserving context
            result[FlowStateKeys.CONVERSATION_HISTORY] = {
                self._component_name: tool_responses + [error_feedback]
            }

            # If we are out of attempts then update execution status to failed
            if (
                attempts + 1 >= self.max_correction_attempts
                and self.execution_result_key
            ):
                status_dict = self.execution_result_key.to_nested_dict("failed")
                result = merge_nested_dict(result, status_dict)
            return result
        else:
            # Success - create success message in conversation_history
            success_message = HumanMessage(
                content=f"Tool execution completed successfully after {attempts} correction attempts."
            )

            # Update conversation_history while preserving context
            result[FlowStateKeys.CONVERSATION_HISTORY] = {
                self._component_name: tool_responses + [success_message]
            }

            # Add success to execution status key
            if self.execution_result_key:
                status_dict = self.execution_result_key.to_nested_dict("success")
                result = merge_nested_dict(result, status_dict)
            return result

    async def _execute_tool(
        self, tool_call_args: dict[str, Any], tool: BaseTool
    ) -> str:
        """Execute a tool with error handling and tracking."""
        try:
            with duo_workflow_metrics.time_tool_call(tool_name=tool.name):
                tool_call_result = await tool.ainvoke(tool_call_args)

            self._track_internal_event(
                event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                tool_name=tool.name,
            )

            self._ui_history.log.success(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
            )

            return tool_call_result
        except Exception as e:
            response = getattr(e, "response", None)
            self._ui_history.log.error(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED,
                tool_response=f"{str(e)} {response}" if response else str(e),
            )
            if isinstance(e, ToolException):
                err_format = self._format_tool_exception(tool_name=tool.name, error=e)
            elif isinstance(e, TypeError):
                err_format = self._format_type_error_response(tool=tool, error=e)
            elif isinstance(e, ValidationError):
                err_format = self._format_validation_error(tool_name=tool.name, error=e)
            else:
                err_format = self._format_execution_error(tool_name=tool.name, error=e)

            return err_format

    def _sanitize_response(
        self,
        response: str | dict | list,
        tool_name: str,
        tool: BaseTool | None = None,
    ) -> str | dict | list:
        """Sanitize tool response for security."""
        try:
            trust_level = getattr(tool, "trust_level", None)
            return apply_security_scanning(
                response=response,
                tool_name=tool_name,
                trust_level=trust_level,
            )
        except SecurityException as e:
            self._logger.error(f"Security validation failed for tool {tool_name}: {e}")
            return (
                f"Security scan blocked the content from tool '{tool_name}'. "
                "The content was flagged as potentially malicious. "
                "Please try a different approach."
            )

    def _track_internal_event(
        self,
        event_name: EventEnum,
        tool_name,
        extra=None,
    ):
        """Track internal events for monitoring."""
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
        """Format type error response for LLM."""
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
        """Format validation error response for LLM."""
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
        """Format execution error response for LLM."""
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        return f"Tool runtime exception due to {str(error)} {getattr(error, "response", None)}"

    def _format_tool_exception(self, tool_name: str, error: ToolException) -> str:
        """Format tool exception response for LLM."""
        self._track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        return f"Tool exception occurred due to {str(error)}"

    def _record_metric(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ) -> None:
        """Record metrics for tool execution."""
        if event_name == EventEnum.WORKFLOW_TOOL_FAILURE:
            tool_name = additional_properties.property or "unknown"
            failure_reason = additional_properties.extra.get("error_type", "unknown")
            duo_workflow_metrics.count_agent_platform_tool_failure(
                flow_type=self._flow_type.value,
                tool_name=tool_name,
                failure_reason=failure_reason,
            )

    def _extract_errors_from_responses(
        self, tool_responses: list[ToolMessage]
    ) -> list[str]:
        """Extract error messages from tool responses."""
        errors = []

        # Regex patterns for our specific error message formats
        error_patterns = [
            r"tool exception occurred due to",  # from _format_tool_exception
            r"execution failed due to",  # from _format_type_error_response
            r"raised validation error",  # from _format_validation_error
            r"runtime exception due to",  # from _format_execution_error
            r"tool \w+ not found",  # from tool not found case - matches "Tool xyz not found"
        ]

        for response in tool_responses:
            content = response.content
            content_to_check = None
            if isinstance(content, str):
                content_to_check = content
            if isinstance(content, list):
                # Convert list content to string for error pattern matching
                content_to_check = " ".join(str(item) for item in content)
            if not content_to_check:
                continue

            # Check for our specific error message formats using regex
            if any(
                re.search(pattern, content_to_check, re.IGNORECASE)
                for pattern in error_patterns
            ):
                errors.append(content_to_check)
        return errors

    def _create_error_feedback(
        self, errors: list[str], tool_calls: list[dict], attempt_count: int
    ) -> HumanMessage:
        """Create detailed error feedback for LLM to correct its mistakes."""

        error_details = []
        for i, error in enumerate(errors):
            if i < len(tool_calls):
                tool_call = tool_calls[i]
                error_details.append(
                    f"Tool call {i+1}: {tool_call['name']}({tool_call.get('args', {})}) "
                    f"failed with error: {error}"
                )
            else:
                error_details.append(f"Error {i+1}: {error}")

        remaining_attempts = self.max_correction_attempts - attempt_count

        feedback_message = (
            f"The previous tool calls failed with the following errors (Attempt {attempt_count}/{self.max_correction_attempts}):\n\n"
            + "\n".join(error_details)
            + f"\n\nYou have {remaining_attempts} attempts remaining. "
            "Please analyze these errors and generate corrected tool calls. "
            "Make sure to:\n"
            "1. Use only tools that exist in the available toolset\n"
            "2. Provide correct argument names and types\n"
            "3. Ensure all required arguments are included\n"
            "4. Validate argument values are appropriate for the tool"
        )

        return HumanMessage(content=feedback_message)
