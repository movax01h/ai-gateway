from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

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
