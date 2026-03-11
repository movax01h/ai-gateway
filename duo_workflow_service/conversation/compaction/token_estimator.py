from langchain_core.messages import AIMessage, BaseMessage

from duo_workflow_service.token_counter.tiktoken_counter import TikTokenCounter


class CompactionTokenEstimator:
    """Token estimation specifically for compaction decisions.

    Uses a hybrid approach:
    - Leverages usage_metadata from AIMessages when available (accurate)
    - Falls back to approximate counting for other messages
    """

    def __init__(self):
        self._counter = TikTokenCounter("compaction")

    def estimate_arbitrary_messages(self, messages: list[BaseMessage]) -> int:
        """Estimate token count for a list of messages.

        Uses actual token counts from AIMessage usage_metadata when available,
        and approximates tokens for other messages using the token counter.

        Args:
            messages: List of messages to estimate tokens for

        Returns:
            Estimated total token count for all messages
        """
        true_tokens = 0
        messages_to_estimate = []

        for msg in messages:
            if (
                isinstance(msg, AIMessage)
                and msg.usage_metadata
                and msg.usage_metadata.get("output_tokens", 0) != 0
            ):
                true_tokens += msg.usage_metadata.get("output_tokens", 0)
            else:
                messages_to_estimate.append(msg)

        return true_tokens + self._counter.count_tokens(
            messages_to_estimate, include_tool_tokens=False
        )

    def estimate_complete_history(self, messages: list[BaseMessage]) -> int:
        """Estimate total token count for complete message history.

        Uses the most recent AIMessage's usage_metadata as a base (if available)
        and counts tokens for trailing messages. Falls back to counting all
        messages if no usage metadata is found.

        Args:
            messages: List of messages to estimate tokens for

        Returns:
            Estimated total token count for the message history
        """
        if not messages:
            return 0

        base_token = 0
        latest_ai_msg_index = 0

        for index, msg in enumerate(reversed(messages)):
            if isinstance(msg, AIMessage) and msg.usage_metadata:
                total_tokens = msg.usage_metadata.get("total_tokens", 0)
                if total_tokens > 0:
                    base_token = total_tokens
                    latest_ai_msg_index = index
                    break

        if base_token == 0:
            return self._counter.count_tokens(messages, include_tool_tokens=False)

        if latest_ai_msg_index == 0:
            return base_token

        trailing_messages = messages[-latest_ai_msg_index:]
        return base_token + self._counter.count_tokens(
            trailing_messages, include_tool_tokens=False
        )
