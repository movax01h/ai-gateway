from collections.abc import Awaitable, Callable

import structlog
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from duo_workflow_service.agents.tools_executor import (
    MALFORMED_TOOL_CALL_ERROR_TEMPLATE,
)
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.tools import MalformedToolCallError, Toolset

log = structlog.stdlib.get_logger("tool_call_validator")

MAX_MALFORMED_TOOL_CALL_RETRIES = 3


def validate_tool_calls(toolset: Toolset, message: AIMessage) -> list[ToolMessage]:
    """Validate tool call arguments against their Pydantic schemas.

    Returns a list of error ToolMessages for any malformed tool calls or an empty list.
    """
    error_messages: list[ToolMessage] = []
    for call in message.tool_calls:
        try:
            toolset.validate_tool_call(call)
        except MalformedToolCallError as e:
            log.warning(
                "Malformed tool call detected, retry LLM request",
                tool_name=call.get("name"),
                tool_call_id=call.get("id"),
                error=str(e),
            )
            error_messages.append(
                ToolMessage(
                    content=str(e),
                    tool_call_id=call.get("id")
                    or f"malformed-{call.get('name', 'unknown')}",
                )
            )

    return error_messages


async def retry_malformed_tool_calls(
    toolset: Toolset,
    agent_response: AIMessage,
    validation_errors: list[ToolMessage],
    state: ChatWorkflowState,
    agent_name: str,
    get_agent_response: Callable[[ChatWorkflowState], Awaitable[BaseMessage]],
) -> BaseMessage:
    """Internally retry LLM calls when tool call arguments are malformed."""
    current_response: BaseMessage = agent_response
    current_errors = validation_errors

    for attempt in range(MAX_MALFORMED_TOOL_CALL_RETRIES):
        log.info(
            "Attempting LLM self-correction for malformed tool calls",
            attempt=attempt + 1,
            max_retries=MAX_MALFORMED_TOOL_CALL_RETRIES,
        )

        # Append the malformed response and error messages to history for the LLM to see
        correction_messages: list[BaseMessage] = [current_response, *current_errors]
        state["conversation_history"][agent_name].extend(correction_messages)

        # Re-invoke the LLM with the updated history
        current_response = await get_agent_response(state)

        # If the new response has no tool calls, or tool calls are now valid, return it
        if (
            not isinstance(current_response, AIMessage)
            or not current_response.tool_calls
        ):
            return current_response

        current_errors = validate_tool_calls(toolset, current_response)
        if not current_errors:
            return current_response

    # All retries exhausted - the LLM keeps producing malformed tool calls
    tool_names = (
        ", ".join(call["name"] for call in getattr(current_response, "tool_calls", []))
        or "unknown"
    )
    log.warning(
        "Max malformed tool call retries exceeded",
        retries=MAX_MALFORMED_TOOL_CALL_RETRIES,
        tool_names=tool_names,
    )
    return AIMessage(
        content=MALFORMED_TOOL_CALL_ERROR_TEMPLATE.format(tool_name=tool_names)
    )
