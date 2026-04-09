import json
from enum import Enum
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, ToolException
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.v1.components.one_off.ui_log import (
    UILogEventsOneOff,
    UILogWriterOneOffTools,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.security.prompt_security import SecurityException
from duo_workflow_service.security.scanner_factory import apply_security_scanning
from duo_workflow_service.tools.toolset import Toolset
from lib.context import client_capabilities
from lib.events import GLReportingEventContext
from lib.hidden_layer_log import set_hidden_layer_log_context
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum, EventLabelEnum

# Routing sentinels used by OneOffComponent._tools_router() to determine next step.
# Defined at module level so they remain accessible even when the class is patched in tests.
SUCCESS_SENTINEL = "completed successfully"
ATTEMPTS_REMAINING_SENTINEL = "attempts remaining"
MAX_ATTEMPTS_SENTINEL = "0 attempts remaining"

# Prefix of the no-tool-calls feedback message; used in tests to assert content.
NO_TOOL_CALLS_FEEDBACK_PREFIX = (
    "Your last response failed to generate the requested tool calls"
)


class ToolExecutionStatus(str, Enum):
    """Status values returned by tool execution."""

    SUCCESS = "success"
    ERROR = "error"


class ToolNodeWithErrorCorrection:  # pylint: disable=too-many-instance-attributes
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
        conversation_history_key: IOKey,
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
        self._conversation_history_key = conversation_history_key

    def _handle_empty_toolset(self, conversation_history: list) -> dict[str, Any]:
        """Handle the case when toolset is empty.

        Args:
            conversation_history: Current conversation history

        Returns:
            State update dict with error message
        """
        result: dict[str, Any] = {**self._ui_history.pop_state_updates()}

        human_message = HumanMessage(
            content=(
                "The agent has no tools configured. "
                f"Review agent privileges configuration. {MAX_ATTEMPTS_SENTINEL}"
            )
        )
        conversation_history_dict = self._conversation_history_key.to_nested_dict(
            conversation_history + [human_message]
        )
        result = merge_nested_dict(result, conversation_history_dict)

        if self.execution_result_key:
            status_dict = self.execution_result_key.to_nested_dict("failed")
            result = merge_nested_dict(result, status_dict)

        return result

    def _handle_no_tool_calls(
        self,
        conversation_history: list,
        context: dict[str, Any],
        context_dict: dict[str, Any],
        attempts: int,
    ) -> dict[str, Any]:
        """Handle the case when no tool calls are present.

        Args:
            conversation_history: Current conversation history
            context: Current context dictionary
            context_dict: Nested context dictionary for merging
            attempts: Current correction attempt count

        Returns:
            State update dict with error feedback
        """
        result: dict[str, Any] = {**self._ui_history.pop_state_updates()}

        feedback_message = (
            f"{NO_TOOL_CALLS_FEEDBACK_PREFIX}. If you were unable to "
            "generate tool calls because you encountered an unresolvable blocker (e.g., authentication "
            "errors, missing credentials, unavailable external resources, or environmental constraints "
            "beyond the scope of available tools), summarize what was completed, what failed, and why — "
            "then stop. Do not attempt further tool calls. If your last response failed to generate tool "
            "calls due to a transient issue (e.g., a formatting error, hallucination, or network "
            "interruption), you MUST retry and generate valid tool calls now to continue your assignment."
        )
        error_feedback = self._create_error_feedback(
            [feedback_message], [], attempts + 1, context
        )

        conversation_history_dict = self._conversation_history_key.to_nested_dict(
            conversation_history + [error_feedback]
        )
        result = merge_nested_dict(result, conversation_history_dict)
        result = merge_nested_dict(result, context_dict)

        # If we are out of attempts then update execution status to failed
        if attempts + 1 >= self.max_correction_attempts and self.execution_result_key:
            status_dict = self.execution_result_key.to_nested_dict("failed")
            result = merge_nested_dict(result, status_dict)

        return result

    async def _execute_all_tool_calls(
        self,
        tool_calls: list[dict],
        conversation_history: list,
        result: dict[str, Any],
    ) -> tuple[list[ToolMessage], dict[str, Any]]:
        """Execute all tool calls and return responses.

        Args:
            tool_calls: List of tool calls to execute
            conversation_history: Current conversation history
            result: Current result dictionary to update

        Returns:
            Tuple of (tool_responses, updated_result)
        """
        tool_responses = []

        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_call_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            if tool_name not in self._toolset:
                response = f"Tool {tool_name} not found"
                status = ToolExecutionStatus.ERROR
            else:
                self._ui_history.log._log_tool_call_input(
                    tool=self._toolset[tool_name],
                    tool_call_args=tool_call_args,
                    event=UILogEventsOneOff.ON_TOOL_CALL_INPUT,
                )

                response, status = await self._execute_tool(
                    tool=self._toolset[tool_name],
                    tool_call_args=tool_call_args,
                )

            tool = self._toolset.get(tool_name)
            set_hidden_layer_log_context(tool_name, tool_call_args)
            sanitized = self._sanitize_response(
                response=response, tool_name=tool_name, tool=tool
            )
            tool_responses.append(
                ToolMessage(
                    content=sanitized,  # type: ignore[arg-type]
                    tool_call_id=tool_call_id,
                    status=status,
                )
            )

        # Build base result with tool responses in conversation history
        conversation_history_dict = self._conversation_history_key.to_nested_dict(
            conversation_history + tool_responses
        )
        result = merge_nested_dict(result, conversation_history_dict)

        # Store tool calls and responses using IOKeys if provided
        if self.tool_calls_key and tool_calls:
            tool_calls_dict = self.tool_calls_key.to_nested_dict(tool_calls)
            result = merge_nested_dict(result, tool_calls_dict)
        if self.tool_responses_key and tool_responses:
            tool_responses_dict = self.tool_responses_key.to_nested_dict(tool_responses)
            result = merge_nested_dict(result, tool_responses_dict)

        return tool_responses, result

    def _handle_tool_execution_errors(
        self,
        errors: list[str],
        tool_calls: list[dict],
        tool_responses: list[ToolMessage],
        conversation_history: list,
        context: dict[str, Any],
        context_dict: dict[str, Any],
        attempts: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle tool execution errors.

        Args:
            errors: List of error messages
            tool_calls: List of tool calls that were executed
            tool_responses: List of tool responses
            conversation_history: Current conversation history
            context: Current context dictionary
            context_dict: Nested context dictionary for merging
            attempts: Current correction attempt count
            result: Current result dictionary to update

        Returns:
            State update dict with error feedback
        """
        # Create error feedback message for LLM
        error_feedback = self._create_error_feedback(
            errors, tool_calls, attempts + 1, context
        )

        # Update conversation_history with error feedback appended
        conversation_history_dict = self._conversation_history_key.to_nested_dict(
            conversation_history + tool_responses + [error_feedback]
        )
        result = merge_nested_dict(result, conversation_history_dict)

        # Update context with correction attempts
        result = merge_nested_dict(result, context_dict)

        # If we are out of attempts then update execution status to failed
        if attempts + 1 >= self.max_correction_attempts and self.execution_result_key:
            status_dict = self.execution_result_key.to_nested_dict("failed")
            result = merge_nested_dict(result, status_dict)

        return result

    def _handle_tool_execution_success(
        self,
        tool_responses: list[ToolMessage],
        conversation_history: list,
        attempts: int,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle successful tool execution.

        Args:
            tool_responses: List of tool responses
            conversation_history: Current conversation history
            attempts: Current correction attempt count
            result: Current result dictionary to update

        Returns:
            State update dict with success message
        """
        # Store parsed responses for deterministic downstream access
        if tool_responses:
            parsed = {}
            for resp in tool_responses:
                try:
                    content = (
                        json.loads(resp.content)
                        if isinstance(resp.content, str)
                        else resp.content
                    )
                    if isinstance(content, dict):
                        parsed.update(content)
                except (json.JSONDecodeError, TypeError):
                    pass
            if parsed:
                parsed_key = IOKey(
                    target="context",
                    subkeys=[self._component_name, "parsed_responses"],
                )
                result = merge_nested_dict(result, parsed_key.to_nested_dict(parsed))

        # Success - create success message in conversation_history
        success_message = HumanMessage(
            content=f"Tool execution {SUCCESS_SENTINEL} after {attempts} correction attempts."
        )

        # Update conversation_history with success message appended
        conversation_history_dict = self._conversation_history_key.to_nested_dict(
            conversation_history + tool_responses + [success_message]
        )
        result = merge_nested_dict(result, conversation_history_dict)

        # Add success to execution status key
        if self.execution_result_key:
            status_dict = self.execution_result_key.to_nested_dict("success")
            result = merge_nested_dict(result, status_dict)

        return result

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Execute tools with error correction tracking."""
        result: dict[str, Any] = {
            **self._ui_history.pop_state_updates(),
        }

        # Get conversation history (needed for replace-mode reducer)
        conversation_history = (
            self._conversation_history_key.value_from_state(state) or []
        )

        if len(self._toolset) == 0:
            return self._handle_empty_toolset(conversation_history)

        # Get current context for error tracking
        context = state.get("context", {}).get(self._component_name, {})
        context_dict = IOKey(
            target="context", subkeys=[self._component_name]
        ).to_nested_dict(context)
        attempts = context.get("correction_attempts", 0)

        # Get tool calls from the last message
        last_message = conversation_history[-1] if conversation_history else None
        tool_calls = getattr(last_message, "tool_calls", []) if last_message else []

        # When LLM decides it cannot continue because of technical/tool limitations
        if not tool_calls:
            return self._handle_no_tool_calls(
                conversation_history, context, context_dict, attempts
            )

        # Execute all tool calls
        tool_responses, result = await self._execute_all_tool_calls(
            tool_calls, conversation_history, result
        )

        # Check for errors in tool responses
        errors = self._extract_errors_from_responses(tool_responses)

        if errors:
            return self._handle_tool_execution_errors(
                errors,
                tool_calls,
                tool_responses,
                conversation_history,
                context,
                context_dict,
                attempts,
                result,
            )

        return self._handle_tool_execution_success(
            tool_responses, conversation_history, attempts, result
        )

    async def _execute_tool(
        self, tool_call_args: dict[str, Any], tool: BaseTool
    ) -> tuple[str, ToolExecutionStatus]:
        """Execute a tool with error handling and tracking.

        Returns:
            tuple[str, ToolExecutionStatus]: (response_content, status)
        """
        try:
            with duo_workflow_metrics.time_tool_call(
                tool_name=tool.name, flow_type=self._flow_type.value
            ):
                tool_call_result = await tool.ainvoke(tool_call_args)

            self._track_internal_event(
                event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
                tool_name=tool.name,
            )

            self._ui_history.log.success(
                tool=tool,
                tool_call_args=tool_call_args,
                event=UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
                tool_response=tool_call_result,
            )

            return tool_call_result, ToolExecutionStatus.SUCCESS
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

            return err_format, ToolExecutionStatus.ERROR

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
            error_message = e.format_user_message(tool_name)
            if tool is not None:
                self._ui_history.log.error(
                    tool=tool,
                    tool_call_args={},
                    message=error_message,
                    event=UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED,
                )
            return error_message

    def _track_internal_event(
        self,
        event_name: EventEnum,
        tool_name,
        extra=None,
    ):
        """Track internal events for monitoring."""
        # Add client capabilities to additional properties
        extra = {
            **(extra or {}),
            "client_capabilities": list(client_capabilities.get()),
        }

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
        """Extract error messages from tool responses by checking status field.

        Returns:
            list[str]: List of error messages from responses with status="error".
        """
        errors = []

        for response in tool_responses:
            if response.status == ToolExecutionStatus.ERROR.value:
                if isinstance(response.content, str):
                    errors.append(response.content)
                elif isinstance(response.content, list):
                    errors.append(" ".join(str(item) for item in response.content))
                else:
                    errors.append(str(response.content))

        return errors

    def _create_error_feedback(
        self,
        errors: list[str],
        tool_calls: list[dict],
        attempt_count: int,
        context: dict[str, Any],
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
            f"The previous tool calls failed with the following errors "
            f"(Attempt {attempt_count}/{self.max_correction_attempts}):\n\n"
            + "\n".join(error_details)
            + f"\n\nYou have {remaining_attempts} {ATTEMPTS_REMAINING_SENTINEL}. "
            "Please analyze these errors and generate corrected tool calls. "
            "Make sure to:\n"
            "1. Use only tools that exist in the available toolset\n"
            "2. Provide correct argument names and types\n"
            "3. Ensure all required arguments are included\n"
            "4. Validate argument values are appropriate for the tool"
        )

        # Update context with correction attempts
        context["correction_attempts"] = attempt_count

        return HumanMessage(content=feedback_message)
