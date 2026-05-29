from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import count_tokens_approximately


class TokenEstimator:
    """Estimate token counts for LangChain messages."""

    def _estimate_arbitrary_messages(self, messages: list[BaseMessage]) -> int:
        """Use real ``output_tokens`` for AIMessages with usage_metadata and approximate the rest."""
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

        return true_tokens + count_tokens_approximately(messages=messages_to_estimate)

    def _estimate_complete_history(self, messages: list[BaseMessage]) -> int:
        """Use the latest ``AIMessage.total_tokens`` as a base and estimate trailing messages.

        Falls back to :meth:`_estimate_arbitrary_messages` when no usable
        cumulative metadata is available.
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
            return self._estimate_arbitrary_messages(messages=messages)

        if latest_ai_msg_index == 0:
            return base_token

        trailing_messages = messages[-latest_ai_msg_index:]
        return base_token + self._estimate_arbitrary_messages(
            messages=trailing_messages
        )

    def count(self, messages: list[BaseMessage], *, is_complete_history: bool) -> int:
        """Estimate the total token count for a list of messages.

        Args:
            messages: List of messages to estimate tokens for.
            is_complete_history: If True, leverage cumulative ``usage_metadata``
                from the latest ``AIMessage`` as a base and only estimate
                trailing messages. If False, estimate each message independently.

        Returns:
            Estimated total token count.
        """
        return (
            self._estimate_complete_history(messages=messages)
            if is_complete_history
            else self._estimate_arbitrary_messages(messages=messages)
        )
