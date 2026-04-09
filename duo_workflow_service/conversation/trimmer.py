import time
from typing import List, cast

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

LEGACY_MAX_CONTEXT_TOKENS = 200_000  # old default for backwards compatibility
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


def _ai_message_tool_call_ids(msg: AIMessage) -> list[str]:
    """Return a deduplicated, ordered list of tool-call IDs declared by an AIMessage."""
    all_calls = msg.tool_calls + msg.invalid_tool_calls
    return list(dict.fromkeys(tc_id for tc in all_calls if (tc_id := tc.get("id"))))


def _fix_tool_call_group(
    ai_msg: AIMessage,
    tool_msgs: list[ToolMessage],
) -> list[BaseMessage]:
    """Fix a single (AIMessage, ToolMessages) group.

    Ensures that every tool_call_id declared by *ai_msg* has exactly one
    ToolMessage immediately following it.  Any ToolMessages whose
    tool_call_id is *not* declared by this AIMessage are treated as
    orphaned and converted to HumanMessages.

    Args:
        ai_msg: The AIMessage that declared tool calls. Must have at least one
            tool_call_id (callers are responsible for this precondition).
        tool_msgs: The ToolMessages that were found immediately after ai_msg.

    Returns:
        A corrected list starting with ai_msg followed by one ToolMessage per
        declared tool_call_id (real where available, synthetic otherwise).
    """
    # list (not set) to preserve declaration order when emitting ToolMessages
    expected_ids: list[str] = _ai_message_tool_call_ids(ai_msg)

    # Index the real ToolMessages by their tool_call_id
    real_by_id: dict[str, ToolMessage] = {}
    orphaned: list[ToolMessage] = []
    for tm in tool_msgs:
        if tm.tool_call_id in expected_ids:
            real_by_id[tm.tool_call_id] = tm
        else:
            orphaned.append(tm)

    missing_ids = [tc_id for tc_id in expected_ids if tc_id not in real_by_id]
    if missing_ids:
        logger.warning(
            "Found AIMessage with unresolved tool_calls, injecting synthetic ToolMessages",
            missing_tool_call_ids=missing_ids,
        )

    result: list[BaseMessage] = [ai_msg]
    for tc_id in expected_ids:
        if tc_id in real_by_id:
            result.append(real_by_id[tc_id])
        else:
            result.append(
                ToolMessage(
                    content=(
                        "Tool execution was interrupted and no result"
                        " is available. Please decide how to proceed."
                    ),
                    tool_call_id=tc_id,
                )
            )

    # Orphaned ToolMessages become HumanMessages
    for tm in orphaned:
        if tm.content:
            result.append(HumanMessage(content=tm.content))

    return result


def restore_message_consistency(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Ensure tool messages and AI messages with tool calls are properly paired.

    Walks through messages in subset groups and fixes each group in place:

    - Single HumanMessage or SystemMessage: passed through unchanged.
    - AIMessage without tool calls: passed through unchanged.
    - AIMessage with tool calls followed by ToolMessages: the immediately
        following ToolMessages are matched against the declared tool_call_ids.
        Missing ToolMessages get a synthetic replacement; ToolMessages whose
        tool_call_id is not declared by the AIMessage are treated as orphaned.
    - Orphaned ToolMessages (not immediately after their AIMessage): converted
        to HumanMessages, because the Anthropic API requires tool_result blocks
        to appear *immediately* after the tool_use block.
    """
    if not messages:
        return []

    result: List[BaseMessage] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and _ai_message_tool_call_ids(msg):
            # Collect the ToolMessages that immediately follow this AIMessage
            i += 1
            tool_msgs: list[ToolMessage] = []
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                tool_msgs.append(cast(ToolMessage, messages[i]))
                i += 1
            result.extend(_fix_tool_call_group(msg, tool_msgs))
        elif isinstance(msg, ToolMessage):
            # Orphaned ToolMessage — not immediately after an AIMessage with tool calls
            if msg.content:
                result.append(HumanMessage(content=msg.content))
            i += 1
        else:
            result.append(msg)
            i += 1

    return result


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


def apply_token_based_trim(
    messages: List[BaseMessage],
    component_name: str,
    max_context_tokens: int,
) -> List[BaseMessage]:
    """Apply token-based trimming to conversation history.

    Only runs when token count exceeds budget. This is expensive due to token counting.

    Note: Message consistency repair (orphaned ToolMessages, dangling AIMessages
    with unresolved tool_calls) is intentionally NOT performed here.  This
    function runs inside the LangGraph state reducer and its output is
    checkpointed — repairing here would persist synthetic ToolMessages into the
    checkpoint state, causing a resume/retry loop.  Consistency is repaired at
    read time in AgentPromptTemplate.invoke and ChatAgentPromptTemplate.invoke.

    Operations:
    1. Replaces oversized single messages with placeholders
    2. Trims conversation to fit budget using LangChain's trim_messages (strategy="last")
    3. Falls back to system + recent messages if trimming fails

    Args:
        messages: List of messages to trim (should be preprocessed first)
        component_name: Name of the component/agent (used for token counting and logging)
        max_context_tokens: Maximum number of tokens allowed in the context window

    Returns:
        Trimmed list of messages that fits within the context window
    """
    if not messages:
        return []

    token_budget = int(TRIM_THRESHOLD * max_context_tokens)
    max_single_message_tokens = int(token_budget * MAX_SINGLE_MESSAGE_TOKEN_SHARE)

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

    t_start = time.perf_counter()
    token_counter = TikTokenCounter(component_name)
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

    return result
