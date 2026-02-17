import time
from typing import List

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter
from duo_workflow_service.tracking.errors import log_exception

logger = structlog.stdlib.get_logger("conversation_trimmer")

LEGACY_MAX_CONTEXT_TOKENS = 400_000  # old default for backwards compatibility
TRIM_THRESHOLD = 0.7  # Only run expensive trimming when utilization exceeds this
MAX_SINGLE_MESSAGE_TOKEN_SHARE = 0.65  # If a single message exceeds this percentage of the context window, pre-trim it


def _find_last_ai_with_usage(
    messages: list[BaseMessage],
) -> tuple[int, int] | None:
    """Find the last AIMessage that has usage_metadata and returns its usage and index."""
    for i, msg in reversed(list(enumerate(messages))):
        if (
            isinstance(msg, AIMessage)
            and msg.usage_metadata
            and msg.usage_metadata.get("total_tokens")
        ):
            return msg.usage_metadata["total_tokens"], i
    return None


def _estimate_tokens_from_history(
    messages: list[BaseMessage], token_counter: TikTokenCounter
) -> int:
    """Estimate total tokens for a full conversation history.

    Uses the cumulative token count from the last AIMessage's usage_metadata
    as a baseline, then estimates only trailing messages (after that AIMessage)
    using TikTokenCounter.

    IMPORTANT: Requires the FULL conversation history. Passing a slice will
    return incorrect results since total_tokens from usage_metadata is cumulative.

    Args:
        messages: Full conversation history as sent to the LLM.
        token_counter: TikTokenCounter instance for counting trailing message tokens.

    Known limitations:
        - The system prompt will always be included in the usage_metadata even if the system message is not
            part of the history.
        - Up to the last AI message with usage metadata we will have accurate counts.
            NOTE: This includes the final rendered prompt templates' content.
            However, the trailing messages will only get a rough estimation based on their content.
            This means, if prompt templates are used (e.g. to render additional_context in message.additional_kwargs)
            based on the message.content, the token count will be underestimated.
    """
    if not messages:
        return 0

    result = _find_last_ai_with_usage(messages)
    if result:
        base_tokens, last_ai_idx = result
        trailing_messages = messages[last_ai_idx + 1 :]
    else:
        base_tokens = 0
        trailing_messages = messages

    trailing_tokens = (
        token_counter.count_tokens(
            trailing_messages,
            # We don't count agent tool tokens - they're included in the AI message's total_tokens already
            include_tool_tokens=False,
        )
        if trailing_messages
        else 0
    )

    return base_tokens + trailing_tokens


def _pretrim_large_messages(
    messages: List[BaseMessage],
    token_counter: TikTokenCounter,
    max_single_message_tokens: int,
) -> List[BaseMessage]:
    """Replace messages that exceed the single message token limit with a placeholder.

    Args:
        messages: List of messages to check
        token_counter: Token counter for the component
        max_single_message_tokens: Maximum tokens allowed for a single message

    Returns:
        List of messages with oversized messages replaced by placeholders
    """
    processed_messages = []
    for message in messages:
        msg_token = token_counter.count_tokens([message], include_tool_tokens=False)
        if msg_token > max_single_message_tokens:
            logger.info(
                f"Message with role: {message.type} token size: {msg_token} "
                f"exceeds the single message token limit: {max_single_message_tokens}. "
                f"Replacing its content with a placeholder."
            )
            message_copy = message.model_copy()
            message_copy.content = (
                "Previous message was too large for context window and was omitted. Please respond "
                "based on the visible context."
            )
            processed_messages.append(message_copy)
        else:
            processed_messages.append(message)
    return processed_messages


def _deduplicate_additional_context(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Remove duplicate <additional_context> tags, keeping only the first occurrence.

    Deduplication is done based on identical content and not ids. If the content changes then the old content and the
    new content will both be kept.
    """
    seen_contexts = set()
    result = []

    for message in messages:
        contexts = message.additional_kwargs.get("additional_context") or []

        new_contexts = []

        for ctx in contexts:
            content = None
            if hasattr(ctx, "content"):
                content = ctx.content
            else:
                # For some reason it's a dict sometimes
                content = ctx.get("content", "")
            if content not in seen_contexts:
                new_contexts.append(ctx)
                seen_contexts.add(content)

        if new_contexts != contexts:
            message_copy = message.model_copy()
            message_copy.additional_kwargs = {
                **message.additional_kwargs,
                "additional_context": new_contexts,
            }
            message = message_copy

        result.append(message)

    return result


def _restore_message_consistency(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Ensure tool messages have corresponding AI messages with tool calls.

    Converts orphaned ToolMessages to HumanMessages to maintain conversation consistency.
    """
    if not messages:
        return []

    # Identify all AIMessages with tool calls
    tool_call_indices = _build_tool_call_indices(messages)

    # Process the messages to ensure consistency
    result: List[BaseMessage] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            # Check if this tool message has a corresponding AIMessage with tool_calls
            # AND if the tool message appears after its parent
            if (
                tool_call_id
                and tool_call_id in tool_call_indices
                and i > tool_call_indices[tool_call_id]
            ):
                result.append(msg)
            else:
                # Convert invalid ToolMessage to HumanMessage
                if msg.content:
                    result.append(HumanMessage(content=msg.content))
        else:
            result.append(msg)

    return result


def _build_tool_call_indices(messages: List[BaseMessage]) -> dict:
    tool_call_indices = {}

    for i, msg in enumerate(messages):
        if not isinstance(msg, AIMessage):
            continue

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_call_indices[tool_call_id] = i

        # Tool calls can end up in `invalid_tool_calls` instead of `tool_calls`
        if hasattr(msg, "invalid_tool_calls") and msg.invalid_tool_calls:
            for invalid_tool_call in msg.invalid_tool_calls:
                tool_call_id = invalid_tool_call.get("id")
                if tool_call_id:
                    tool_call_indices[tool_call_id] = i

    return tool_call_indices


def _get_message_roles(messages: List[BaseMessage]) -> List[str]:
    """Get the roles for a list of messages."""
    return [msg.type for msg in messages]


def _fallback_messages(
    messages: List[BaseMessage], min_recent: int = 3
) -> List[BaseMessage]:
    """Build a minimal fallback: all system messages + the last `min_recent` non-system messages."""
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    non_system_messages = [
        msg for msg in messages if not isinstance(msg, SystemMessage)
    ]
    n = min(min_recent, len(non_system_messages))
    return system_messages + non_system_messages[-n:]


def trim_conversation_history(
    messages: List[BaseMessage],
    component_name: str,
    max_context_tokens: int,
) -> List[BaseMessage]:
    """Trim conversation history to fit within the model's context window.

    Always runs (cheap, no token counting):
    1. Deduplicates additional context tags
    2. Restores message consistency (converts orphaned tool messages to human messages)

    Only when token count exceeds the token budget to avoid unnecessary computation (token counting is expensive):
    3. Replaces oversized single messages with placeholders
    4. Trims conversation to fit budget using LangChain's trim_messages (strategy="last")
    5. Falls back to system + recent messages if trimming produces empty/invalid results

    Args:
        messages: List of messages to trim
        component_name: Name of the component/agent (used for token counting and logging)
        max_context_tokens: Maximum number of tokens allowed in the context window

    Returns:
        Trimmed list of messages that fits within the context window
    """
    if not messages:
        return []

    token_budget = int(TRIM_THRESHOLD * max_context_tokens)
    max_single_message_tokens = int(token_budget * MAX_SINGLE_MESSAGE_TOKEN_SHARE)

    # Always run: cheap maintenance (no token counting)
    messages = _deduplicate_additional_context(messages)

    token_counter = TikTokenCounter(component_name)
    initial_tokens = _estimate_tokens_from_history(
        messages=messages, token_counter=token_counter
    )
    initial_roles = _get_message_roles(messages)

    if initial_tokens < token_budget:
        logger.info(
            "Skipping trimming since under budget threshold.",
            component_name=component_name,
            message_roles=initial_roles,
            current_tokens=initial_tokens,
            max_context_tokens=max_context_tokens,
            token_budget=token_budget,
            budget_utilization_pct=(
                round(initial_tokens / token_budget * 100, 1) if token_budget > 0 else 0
            ),
        )
        return messages

    # Costly trimming pipeline — only runs when over budget
    t_start = time.perf_counter()
    # Use TikTokenCounter for trimming - it supports slicing which is needed
    # by LangChain's trim_messages binary search
    token_counter = TikTokenCounter(component_name)
    # We count once more to be able to compare before and after counts derived from the same method
    initial_tokens = token_counter.count_tokens(messages, include_tool_tokens=True)

    logger.info(
        "Starting trimming",
        component_name=component_name,
        message_roles=initial_roles,
        initial_tokens=initial_tokens,
        max_context_tokens=max_context_tokens,
        token_budget=token_budget,
        budget_utilization_pct=round(initial_tokens / token_budget * 100, 1),
    )

    processed_messages = _pretrim_large_messages(
        messages, token_counter, max_single_message_tokens
    )

    try:
        result = trim_messages(
            processed_messages,
            max_tokens=token_budget,
            strategy="last",
            token_counter=token_counter.count_tokens,
            start_on="human",
            include_system=True,
            allow_partial=False,
        )

        if not result or (len(result) == 1 and isinstance(result[0], SystemMessage)):
            logger.warning(
                "Trim resulted in empty/invalid messages - falling back to minimal context",
                component_name=component_name,
            )
            result = _fallback_messages(messages, min_recent=3)

    except Exception as e:
        log_exception(
            e,
            extra={
                "context": "Error during message trimming",
                "component_name": component_name,
            },
        )
        logger.warning(
            "Exception during trimming - falling back to minimal context",
            component_name=component_name,
        )
        result = _fallback_messages(messages, min_recent=5)

    duration_ms = round((time.perf_counter() - t_start) * 1000, 2)
    final_roles = _get_message_roles(result)
    final_tokens = token_counter.count_tokens(result, include_tool_tokens=True)

    logger.info(
        "Finished trimming",
        message_roles=final_roles,
        component_name=component_name,
        initial_tokens=initial_tokens,
        final_tokens=final_tokens,
        max_context_tokens=max_context_tokens,
        token_budget=token_budget,
        budget_utilization_pct=(
            round(final_tokens / token_budget * 100, 1) if token_budget > 0 else 0
        ),
        duration_ms=duration_ms,
    )

    return _restore_message_consistency(result)
