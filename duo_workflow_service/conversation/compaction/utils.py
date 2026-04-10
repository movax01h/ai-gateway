import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from duo_workflow_service.conversation.compaction.schema import (
    CompactionConfig,
    MessageSlices,
)
from duo_workflow_service.conversation.compaction.token_estimator import (
    CompactionTokenEstimator,
)


def is_turn_complete(messages: list[BaseMessage]) -> bool:
    """Determine if a conversation is at the end of a complete turn.

    A complete turn means:
    - Latest message is HumanMessage, or
    - Latest message is AIMessage without tool calls, or
    - Latest message is ToolMessage with all tool calls resolved

    Args:
        messages: List of LangChain messages representing the conversation history

    Returns:
        True if the conversation is at the end of a complete turn, False otherwise
    """
    if not messages:
        return True

    latest = messages[-1]

    if isinstance(latest, HumanMessage):
        return True

    if isinstance(latest, AIMessage):
        return not latest.tool_calls and not latest.invalid_tool_calls

    if isinstance(latest, ToolMessage):
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                ai_tool_call_ids = {
                    tc["id"] for tc in msg.tool_calls + msg.invalid_tool_calls
                }
                tool_message_ids = {
                    m.tool_call_id
                    for m in messages[i + 1 :]
                    if isinstance(m, ToolMessage)
                }
                return ai_tool_call_ids == tool_message_ids

        return False

    return False


def resolve_recent_messages_internal(
    messages: list[BaseMessage],
    config: CompactionConfig,
    token_estimator: CompactionTokenEstimator,
) -> list[BaseMessage]:
    """Collect complete turns from the end, respecting token and message count limits.

    Internal function used by compactor. For external use, prefer the wrapper
    in the compaction package __init__.py.

    Args:
        messages: List of messages to collect from
        config: Compaction configuration with limits
        token_estimator: Token estimator for counting

    Returns:
        Recent messages in correct order
    """
    if not messages:
        return []

    recent_messages: list[BaseMessage] = []
    current_turn: list[BaseMessage] = []
    total_tokens = 0

    for msg in reversed(messages):
        current_turn.append(msg)
        turn = current_turn[::-1]

        if is_turn_complete(turn):
            turn_tokens = token_estimator.estimate_arbitrary_messages(turn)
            if (
                total_tokens + turn_tokens > config.recent_messages_token_budget
                or len(turn + recent_messages) > config.max_recent_messages
            ):
                break

            recent_messages = turn + recent_messages
            total_tokens += turn_tokens
            current_turn = []

    return recent_messages


def slice_for_summarization(
    messages: list[BaseMessage],
    config: CompactionConfig,
    token_estimator: CompactionTokenEstimator,
) -> MessageSlices:
    """Split messages into three parts for summarization.

    Args:
        messages: List of messages to slice
        config: Compaction configuration
        token_estimator: Token estimator for counting

    Returns:
        MessageSlices with leading_context, to_summarize, and recent_to_keep
    """
    if not messages:
        return MessageSlices(leading_context=[], to_summarize=[], recent_to_keep=[])

    leading_context_end = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            leading_context_end = i + 1
        else:
            break

    leading_context = messages[:leading_context_end]
    remaining = messages[leading_context_end:]

    recent_to_keep = resolve_recent_messages_internal(
        remaining, config, token_estimator
    )
    to_summarize = remaining[: len(remaining) - len(recent_to_keep)]

    return MessageSlices(
        leading_context=leading_context,
        to_summarize=to_summarize,
        recent_to_keep=recent_to_keep,
    )


def _format_tool_calls_as_text(
    tool_calls: list[ToolCall],
) -> str:
    """Format tool_calls into a human-readable string representation.

    Preserves the original tool call structure (name, arguments) in text form so the summarizer can understand what
    actions the assistant took.
    """
    parts = []
    for tc in tool_calls:
        name = tc.get("name", "unknown")
        args = tc.get("args", {})
        args_str = json.dumps(args, indent=2) if args else "{}"
        parts.append(f"[Called tool '{name}' with arguments: {args_str}]")
    return "\n".join(parts)


def strip_tool_metadata_for_litellm(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Convert tool-calling metadata to text representations for summarization.

    Workaround for LiteLLM bug https://github.com/BerriAI/litellm/issues/24712
    where LiteLLM raises UnsupportedParamsError when messages contain tool_calls
    but no tools= param is specified. This only affects LiteLLM-backed providers
    (e.g., Vertex AI). Once the upstream bug is fixed, this function should be
    removed.

    Transforms:
    - AIMessage with tool_calls -> AIMessage with text content + tool calls as text
    - ToolMessage -> HumanMessage with tool result as text
    - All other messages -> passed through unchanged
    """
    cleaned: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Extract text content from the message
            content = msg.content
            if isinstance(content, list):
                # Filter out tool_use blocks, keep text blocks
                text_blocks = [
                    block
                    for block in content
                    if not (isinstance(block, dict) and block.get("type") == "tool_use")
                ]
                if (
                    len(text_blocks) == 1
                    and isinstance(text_blocks[0], dict)
                    and text_blocks[0].get("type") == "text"
                ):
                    content = text_blocks[0].get("text", "")
                elif not text_blocks:
                    content = ""
                else:
                    content = text_blocks

            # Append tool calls as text so the summarizer sees what actions were taken
            tool_calls_text = _format_tool_calls_as_text(msg.tool_calls)
            if isinstance(content, str):
                content = (
                    f"{content}\n{tool_calls_text}".strip()
                    if content
                    else tool_calls_text
                )
            # For list content, create a new list to avoid mutating the original
            elif isinstance(content, list):
                content = [*content, {"type": "text", "text": tool_calls_text}]

            cleaned.append(AIMessage(content=content))
        elif isinstance(msg, ToolMessage):
            # Convert tool result to a human-readable message
            tool_name = msg.name or "unknown"
            cleaned.append(
                HumanMessage(content=f"[Tool result for '{tool_name}']: {msg.content}")
            )
        else:
            cleaned.append(msg)
    return cleaned
