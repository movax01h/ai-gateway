from typing import List, Tuple

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
                f"exceeds the single message token limit: {max_single_message_tokens}."
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
    tool_call_indices = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_call_indices[tool_call_id] = i

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


def get_messages_profile(
    messages: List[BaseMessage],
    token_counter: TikTokenCounter,
    include_tool_tokens: bool = True,
) -> Tuple[List[str], int]:
    """Get the roles and token count for a list of messages.

    Args:
        messages: List of messages to profile
        token_counter: Token counter for the component
        include_tool_tokens: Whether to include tool specification tokens in the count

    Returns:
        Tuple of (list of message roles, total token count)
    """
    roles = [msg.type for msg in messages]
    token_size = (
        token_counter.count_tokens(messages, include_tool_tokens=include_tool_tokens)
        if messages
        else 0
    )
    return roles, token_size


def trim_conversation_history(
    messages: List[BaseMessage],
    component_name: str,
    max_context_tokens: int,
) -> List[BaseMessage]:
    """Trim conversation history to fit within the model's context window.

    This function performs the following steps:
    1. Pre-trims oversized single messages
    2. Deduplicates additional context
    3. Trims the conversation to max_context_tokens using LangChain's trim_messages
    4. Restores message consistency (ensures tool messages have corresponding AI messages)
    5. Falls back to minimal context if trimming results in empty messages

    Args:
        messages: List of messages to trim
        max_context_tokens: Maximum number of tokens allowed in the context window
        component_name: Name of the component/agent (used for token counting and logging)

    Returns:
        Trimmed list of messages that fits within the context window
    """
    if not messages:
        return []

    max_context_tokens = int(0.7 * max_context_tokens)

    token_counter = TikTokenCounter(component_name)
    max_single_message_tokens = int(max_context_tokens * 0.65)

    # Log initial state
    initial_roles, initial_tokens = get_messages_profile(
        messages=messages,
        token_counter=token_counter,
        include_tool_tokens=False,
    )

    logger.info(
        f"Starting trimming for {component_name} with "
        f"messages roles: {initial_roles}, token size: {initial_tokens}, "
        f"max_context_tokens: {max_context_tokens}",
        component_name=component_name,
        initial_tokens=initial_tokens,
        max_context_tokens=max_context_tokens,
    )

    processed_messages = _pretrim_large_messages(
        messages, token_counter, max_single_message_tokens
    )

    deduplicated_messages = _deduplicate_additional_context(processed_messages)

    try:
        trimmed_messages = trim_messages(
            deduplicated_messages,
            max_tokens=max_context_tokens,
            strategy="last",
            token_counter=token_counter.count_tokens,
            start_on="human",
            include_system=True,
            allow_partial=False,
        )

        result = _restore_message_consistency(trimmed_messages)

        # Step 5: Fallback if trimming resulted in empty or invalid messages
        if not result or (len(result) == 1 and isinstance(result[0], SystemMessage)):
            logger.warning(
                "Trim resulted in empty messages/invalid messages - falling back to minimal context",
                component_name=component_name,
            )

            system_messages = [
                msg for msg in messages if isinstance(msg, SystemMessage)
            ]
            non_system_messages = [
                msg for msg in messages if not isinstance(msg, SystemMessage)
            ]

            min_non_system = min(3, len(non_system_messages))
            fallback_messages = system_messages + non_system_messages[-min_non_system:]

            result = _restore_message_consistency(fallback_messages)

        # Log final state
        final_roles, final_tokens = get_messages_profile(
            messages=result,
            token_counter=token_counter,
            include_tool_tokens=False,
        )

        logger.info(
            f"Finished trimming for {component_name} with "
            f"messages roles: {final_roles}, token size: {final_tokens}",
            component_name=component_name,
            initial_tokens=initial_tokens,
            final_tokens=final_tokens,
            max_context_tokens=max_context_tokens,
        )

        return result

    except Exception as e:
        log_exception(
            e,
            extra={
                "context": "Error during message trimming",
                "component_name": component_name,
            },
        )

        # Fallback: Keep system messages plus a few recent messages
        logger.warning(
            "Exception during trimming - falling back to minimal context",
            component_name=component_name,
        )

        system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
        non_system_messages = [
            msg for msg in messages if not isinstance(msg, SystemMessage)
        ]

        fallback_messages = system_messages + non_system_messages[-5:]
        return _restore_message_consistency(fallback_messages)
