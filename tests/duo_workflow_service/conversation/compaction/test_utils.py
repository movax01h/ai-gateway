# pylint: disable=too-many-lines
"""Tests for conversation compaction utility functions."""

from unittest.mock import patch

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    CompactionTokenEstimator,
    is_turn_complete,
    resolve_recent_messages_internal,
    slice_for_summarization,
)
from duo_workflow_service.conversation.compaction.utils import (
    _format_tool_calls_as_text,
    strip_tool_metadata_for_litellm,
)


class TestIsTurnComplete:
    """Test suite for is_turn_complete function."""

    def test_empty_messages_returns_true(self):
        """Empty message list should be considered a complete turn."""
        assert is_turn_complete([]) is True

    def test_single_human_message_is_complete(self):
        """A single HumanMessage should be a complete turn."""
        messages = [HumanMessage(content="Hello")]
        assert is_turn_complete(messages) is True

    def test_multiple_human_messages_is_complete(self):
        """Multiple HumanMessages ending with HumanMessage should be complete."""
        messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
        ]
        assert is_turn_complete(messages) is True

    def test_ai_message_without_tool_calls_is_complete(self):
        """AIMessage without tool calls should be a complete turn."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Response"),
        ]
        assert is_turn_complete(messages) is True

    def test_ai_message_with_tool_calls_is_incomplete(self):
        """AIMessage with tool calls should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="I'll use a tool",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
        ]
        assert is_turn_complete(messages) is False

    def test_ai_message_with_invalid_tool_calls_is_incomplete(self):
        """AIMessage with invalid tool calls should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Invalid tool call",
                invalid_tool_calls=[{"id": "invalid_1", "name": "bad_tool"}],
            ),
        ]
        assert is_turn_complete(messages) is False

    def test_ai_message_with_both_tool_calls_and_invalid_is_incomplete(self):
        """AIMessage with both tool calls and invalid tool calls should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Mixed tool calls",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
                invalid_tool_calls=[{"id": "invalid_1", "name": "bad_tool"}],
            ),
        ]
        assert is_turn_complete(messages) is False

    def test_tool_message_with_matching_tool_call_is_complete(self):
        """ToolMessage with matching AIMessage tool call should be complete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="I'll search",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
        ]
        assert is_turn_complete(messages) is True

    def test_tool_message_with_multiple_matching_tool_calls_is_complete(self):
        """ToolMessages matching all AIMessage tool calls should be complete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="I'll use multiple tools",
                tool_calls=[
                    {"id": "tool_1", "name": "search", "args": {}},
                    {"id": "tool_2", "name": "fetch", "args": {}},
                ],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            ToolMessage(content="Fetch result", tool_call_id="tool_2"),
        ]
        assert is_turn_complete(messages) is True

    def test_tool_message_with_missing_tool_call_is_incomplete(self):
        """ToolMessage when AIMessage has more tool calls should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="I'll use multiple tools",
                tool_calls=[
                    {"id": "tool_1", "name": "search", "args": {}},
                    {"id": "tool_2", "name": "fetch", "args": {}},
                ],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
        ]
        assert is_turn_complete(messages) is False

    def test_tool_message_without_matching_ai_message_is_incomplete(self):
        """ToolMessage without corresponding AIMessage should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            ToolMessage(content="Orphaned tool result", tool_call_id="tool_1"),
        ]
        assert is_turn_complete(messages) is False

    def test_tool_message_with_wrong_tool_call_id_is_incomplete(self):
        """ToolMessage without tool_call_id should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="I'll search",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="wrong_id"),
        ]
        assert is_turn_complete(messages) is False

    def test_multiple_tool_messages_with_extra_tool_message_is_incomplete(self):
        """Extra ToolMessage not matching any tool call should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="I'll search",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            ToolMessage(content="Extra result", tool_call_id="tool_2"),
        ]
        assert is_turn_complete(messages) is False

    def test_complex_conversation_ending_with_human_is_complete(self):
        """Complex conversation ending with HumanMessage should be complete."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(
                content="I'll search for that",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            AIMessage(content="Here's the answer"),
            HumanMessage(content="Follow-up question"),
        ]
        assert is_turn_complete(messages) is True

    def test_complex_conversation_ending_with_ai_no_tools_is_complete(self):
        """Complex conversation ending with AIMessage without tools should be complete."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(
                content="I'll search for that",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            AIMessage(content="Here's the answer"),
        ]
        assert is_turn_complete(messages) is True

    def test_complex_conversation_ending_with_ai_with_tools_is_incomplete(self):
        """Complex conversation ending with AIMessage with tools should be incomplete."""
        messages = [
            HumanMessage(content="First question"),
            AIMessage(
                content="I'll search for that",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            AIMessage(
                content="I need more info",
                tool_calls=[{"id": "tool_2", "name": "fetch", "args": {}}],
            ),
        ]
        assert is_turn_complete(messages) is False

    def test_tool_message_with_invalid_tool_calls_in_ai_message(self):
        """ToolMessage matching invalid tool call in AIMessage should be complete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Invalid tool call",
                invalid_tool_calls=[{"id": "invalid_1", "name": "bad_tool"}],
            ),
            ToolMessage(content="Tool result", tool_call_id="invalid_1"),
        ]
        assert is_turn_complete(messages) is True

    def test_tool_message_with_mixed_tool_calls_in_ai_message(self):
        """ToolMessages matching both tool_calls and invalid_tool_calls should be complete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Mixed tool calls",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
                invalid_tool_calls=[{"id": "invalid_1", "name": "bad_tool"}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            ToolMessage(content="Invalid result", tool_call_id="invalid_1"),
        ]
        assert is_turn_complete(messages) is True

    def test_tool_message_with_mixed_tool_calls_missing_one(self):
        """ToolMessages missing one of mixed tool calls should be incomplete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Mixed tool calls",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
                invalid_tool_calls=[{"id": "invalid_1", "name": "bad_tool"}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
        ]
        assert is_turn_complete(messages) is False

    def test_multiple_ai_messages_with_tool_calls_only_last_matters(self):
        """Only the last message matters for determining turn completion."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="First response with tools",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            AIMessage(
                content="Second response with tools",
                tool_calls=[{"id": "tool_2", "name": "fetch", "args": {}}],
            ),
        ]
        assert is_turn_complete(messages) is False

    def test_ai_message_with_empty_tool_calls_list_is_complete(self):
        """AIMessage with empty tool_calls list should be complete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Response", tool_calls=[]),
        ]
        assert is_turn_complete(messages) is True

    def test_ai_message_with_empty_invalid_tool_calls_list_is_complete(self):
        """AIMessage with empty invalid_tool_calls list should be complete."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Response", invalid_tool_calls=[]),
        ]
        assert is_turn_complete(messages) is True


def _mock_estimate_tokens(messages: list[BaseMessage]) -> int:
    """Mock token estimation using 4 bytes per token approximation."""
    total_bytes = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else ""
        if isinstance(content, str):
            total_bytes += len(content.encode("utf-8"))
    return total_bytes // 4


@patch.object(
    CompactionTokenEstimator,
    "estimate_arbitrary_messages",
    side_effect=_mock_estimate_tokens,
)
class TestResolveRecentMessagesInternal:
    """Test suite for resolve_recent_messages_internal function."""

    mock_chat_messages = [
        HumanMessage(content="This is my first query"),
        AIMessage(
            content="I'll use tool x",
            tool_calls=[
                {"id": "tool_1", "name": "search", "args": {}},
            ],
        ),
        ToolMessage(content="Search result", tool_call_id="tool_1"),
        AIMessage(
            content="I'll use multiple tools",
            tool_calls=[
                {"id": "tool_1", "name": "search", "args": {}},
                {"id": "tool_2", "name": "fetch", "args": {}},
            ],
        ),
        ToolMessage(content="Search result", tool_call_id="tool_1"),
        ToolMessage(content="Fetch result", tool_call_id="tool_2"),
        AIMessage(content="Here is the answers for your response"),
        HumanMessage(content="New query"),
    ]

    mock_auto_messages = [
        HumanMessage(content="Help me implement x feature"),
        AIMessage(
            content="I'll use multiple tools",
            tool_calls=[
                {"id": "tool_1", "name": "search", "args": {}},
                {"id": "tool_2", "name": "fetch", "args": {}},
            ],
        ),
        ToolMessage(content="Search result", tool_call_id="tool_1"),
        ToolMessage(content="Fetch result", tool_call_id="tool_2"),
        AIMessage(
            content="Now I understand x. I'll use tool y",
            tool_calls=[
                {"id": "tool_1", "name": "grep", "args": {}},
            ],
        ),
        ToolMessage(content="Grep result", tool_call_id="tool_1"),
        AIMessage(
            content="Now, let me understand the test structure. I'll use multiple tools",
            tool_calls=[
                {"id": "tool_1", "name": "read", "args": {}},
                {"id": "tool_2", "name": "grep", "args": {}},
                {"id": "tool_3", "name": "grep", "args": {}},
            ],
        ),
        ToolMessage(content="Read result", tool_call_id="tool_1"),
        ToolMessage(content="Grep result", tool_call_id="tool_2"),
        ToolMessage(content="Grep result", tool_call_id="tool_3"),
        AIMessage(
            content="Now let me implement it.",
            tool_calls=[
                {"id": "tool_4", "name": "edit", "args": {}},
            ],
        ),
        ToolMessage(content="Edit result", tool_call_id="tool_4"),
        AIMessage(
            content="I have implemented the feature for you, here is the summary..."
        ),
    ]

    def test_empty_messages_returns_empty(self, mock_estimate):
        """Empty message list should return empty."""
        config = CompactionConfig()
        estimator = CompactionTokenEstimator()
        assert resolve_recent_messages_internal([], config, estimator) == []
        mock_estimate.assert_not_called()

    def test_with_chat_style_history_with_length_limit(self, mock_estimate):
        """Test with chat-style history respecting max_recent_messages limit."""
        messages = self.mock_chat_messages
        estimator = CompactionTokenEstimator()

        for i in [1, 2, 7, 8]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[-i:]

        for i in [3, 4]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == [
                AIMessage(content="Here is the answers for your response"),
                HumanMessage(content="New query"),
            ]

        for i in [5, 6]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == [
                AIMessage(
                    content="I'll use multiple tools",
                    tool_calls=[
                        {"id": "tool_1", "name": "search", "args": {}},
                        {"id": "tool_2", "name": "fetch", "args": {}},
                    ],
                ),
                ToolMessage(content="Search result", tool_call_id="tool_1"),
                ToolMessage(content="Fetch result", tool_call_id="tool_2"),
                AIMessage(content="Here is the answers for your response"),
                HumanMessage(content="New query"),
            ]

        assert mock_estimate.called

    def test_with_auto_mode_history_with_length_limit(self, mock_estimate):
        """Test with auto-mode history respecting max_recent_messages limit."""
        messages = self.mock_auto_messages
        estimator = CompactionTokenEstimator()

        for i in [1, 2]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[-1:]

        for i in [3, 4, 5, 6]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[-3:]

        for i in [7, 8]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[-7:]

        for i in [9, 10, 11]:
            config = CompactionConfig(
                max_recent_messages=i, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[-9:]

        for i in [12, 13, 14]:
            config = CompactionConfig(
                max_recent_messages=12, recent_messages_token_budget=100_000
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[-12:]

        assert mock_estimate.called

    def test_with_chat_style_history_with_token_limit(self, mock_estimate):
        """Test with chat-style history respecting token budget limit."""
        messages = self.mock_chat_messages
        estimator = CompactionTokenEstimator()

        config = CompactionConfig(
            max_recent_messages=100, recent_messages_token_budget=1
        )
        result = resolve_recent_messages_internal(messages, config, estimator)
        assert result == []

        for limit in [2, 4, 7, 10]:
            config = CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=limit
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == [HumanMessage(content="New query")]

        for limit in [11, 15, 19, 22]:
            config = CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=limit
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == [
                AIMessage(content="Here is the answers for your response"),
                HumanMessage(content="New query"),
            ]

        for limit in [23, 26, 29]:
            config = CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=limit
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == [
                AIMessage(
                    content="I'll use multiple tools",
                    tool_calls=[
                        {"id": "tool_1", "name": "search", "args": {}},
                        {"id": "tool_2", "name": "fetch", "args": {}},
                    ],
                ),
                ToolMessage(content="Search result", tool_call_id="tool_1"),
                ToolMessage(content="Fetch result", tool_call_id="tool_2"),
                AIMessage(content="Here is the answers for your response"),
                HumanMessage(content="New query"),
            ]

        for limit in [30, 33, 34]:
            config = CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=limit
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages[1:]

        for limit in [35, 37]:
            config = CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=limit
            )
            result = resolve_recent_messages_internal(messages, config, estimator)
            assert result == messages

        assert mock_estimate.called


@patch.object(CompactionTokenEstimator, "estimate_arbitrary_messages", return_value=10)
class TestSliceForSummarization:
    """Test suite for slice_for_summarization function."""

    def test_empty_messages(self, mock_estimate):
        """Should return empty slices for empty input."""
        config = CompactionConfig()
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization([], config, estimator)

        assert result.leading_context == []
        assert result.to_summarize == []
        assert result.recent_to_keep == []

        mock_estimate.assert_not_called()

    def test_only_leading_human_messages(self, mock_estimate):
        """Should preserve leading HumanMessages in leading_context."""
        messages = [
            HumanMessage(content="first"),
            HumanMessage(content="second"),
            AIMessage(content="response"),
        ]
        config = CompactionConfig(
            max_recent_messages=1, recent_messages_token_budget=100_000
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="first"),
            HumanMessage(content="second"),
        ]
        assert result.to_summarize == []
        assert result.recent_to_keep == [AIMessage(content="response")]
        mock_estimate.assert_called()

    def test_no_leading_context(self, mock_estimate):
        """Should handle messages starting with AIMessage."""
        messages = [
            AIMessage(content="response1"),
            HumanMessage(content="query"),
            AIMessage(content="response2"),
        ]
        config = CompactionConfig(
            max_recent_messages=1, recent_messages_token_budget=100_000
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == []
        assert result.to_summarize == [
            AIMessage(content="response1"),
            HumanMessage(content="query"),
        ]
        assert result.recent_to_keep == [AIMessage(content="response2")]
        mock_estimate.assert_called()

    def test_with_mixed_messages(self, mock_estimate):
        """Should correctly split leading, to_summarize, and recent."""
        messages = [
            HumanMessage(content="initial query"),
            AIMessage(content="first response"),
            HumanMessage(content="follow up"),
            AIMessage(content="second response"),
            HumanMessage(content="final query"),
        ]
        config = CompactionConfig(
            max_recent_messages=2, recent_messages_token_budget=100_000
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="initial query"),
        ]
        assert result.to_summarize == [
            AIMessage(content="first response"),
            HumanMessage(content="follow up"),
        ]
        assert result.recent_to_keep == [
            AIMessage(content="second response"),
            HumanMessage(content="final query"),
        ]
        mock_estimate.assert_called()

    def test_respects_config_max_recent_messages(self, mock_estimate):
        """Should respect max_recent_messages from config."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="response1"),
            AIMessage(content="response2"),
            AIMessage(content="response3"),
        ]
        config = CompactionConfig(
            max_recent_messages=2, recent_messages_token_budget=100_000
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="query"),
        ]
        assert result.to_summarize == [
            AIMessage(content="response1"),
        ]
        assert result.recent_to_keep == [
            AIMessage(content="response2"),
            AIMessage(content="response3"),
        ]
        mock_estimate.assert_called()


@patch.object(
    CompactionTokenEstimator,
    "estimate_arbitrary_messages",
    side_effect=_mock_estimate_tokens,
)
class TestSliceForSummarizationWithTokenBudget:
    """Test suite for slice_for_summarization with token budget limits.

    Token calculation: len(content.encode('utf-8')) // 4
    Example message tokens (for reference):
    - "initial query" = 3 tokens
    - "first response" = 3 tokens
    - "follow up question" = 4 tokens
    - "second response" = 3 tokens
    - "final query" = 2 tokens
    """

    def test_small_token_budget_keeps_fewer_recent(self, mock_estimate):
        """Should keep fewer recent messages when token budget is small.

        Budget=5: can fit "final query"(2) + "second response"(3) = 5 tokens
        """
        messages = [
            HumanMessage(content="initial query"),
            AIMessage(content="first response"),
            HumanMessage(content="follow up question"),
            AIMessage(content="second response"),
            HumanMessage(content="final query"),
        ]
        config = CompactionConfig(
            max_recent_messages=100, recent_messages_token_budget=5
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="initial query"),
        ]
        assert result.to_summarize == [
            AIMessage(content="first response"),
            HumanMessage(content="follow up question"),
        ]
        assert result.recent_to_keep == [
            AIMessage(content="second response"),
            HumanMessage(content="final query"),
        ]
        mock_estimate.assert_called()

    def test_medium_token_budget_keeps_more_recent(self, mock_estimate):
        """Should keep more recent messages when token budget is medium.

        Budget=9: can fit "final query"(2) + "second response"(3) + "follow up question"(4) = 9 tokens
        """
        messages = [
            HumanMessage(content="initial query"),
            AIMessage(content="first response"),
            HumanMessage(content="follow up question"),
            AIMessage(content="second response"),
            HumanMessage(content="final query"),
        ]
        config = CompactionConfig(
            max_recent_messages=100, recent_messages_token_budget=9
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="initial query"),
        ]
        assert result.to_summarize == [
            AIMessage(content="first response"),
        ]
        assert result.recent_to_keep == [
            HumanMessage(content="follow up question"),
            AIMessage(content="second response"),
            HumanMessage(content="final query"),
        ]
        mock_estimate.assert_called()

    def test_large_token_budget_keeps_all_recent(self, mock_estimate):
        """Should keep all messages as recent when token budget is large."""
        messages = [
            HumanMessage(content="initial query"),
            AIMessage(content="first response"),
            HumanMessage(content="follow up question"),
            AIMessage(content="second response"),
        ]
        config = CompactionConfig(
            max_recent_messages=100, recent_messages_token_budget=100_000
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="initial query"),
        ]
        assert result.to_summarize == []
        assert result.recent_to_keep == [
            AIMessage(content="first response"),
            HumanMessage(content="follow up question"),
            AIMessage(content="second response"),
        ]
        mock_estimate.assert_called()

    def test_minimal_token_budget_keeps_nothing(self, mock_estimate):
        """Should keep no recent messages when token budget is minimal."""
        messages = [
            HumanMessage(content="initial query"),
            AIMessage(content="first response"),
            HumanMessage(content="follow up"),
        ]
        config = CompactionConfig(
            max_recent_messages=100, recent_messages_token_budget=1
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="initial query"),
        ]
        assert result.to_summarize == [
            AIMessage(content="first response"),
            HumanMessage(content="follow up"),
        ]
        assert result.recent_to_keep == []
        mock_estimate.assert_called()

    def test_token_budget_with_tool_messages(self, mock_estimate):
        """Should respect token budget when messages include tool calls.

        Token counts:
        - "query" = 1 token
        - "I'll search" = 2 tokens
        - "Search result" = 3 tokens
        - "Here is the answer" = 4 tokens
        - "thanks" = 1 token

        Budget=5: can fit "thanks"(1) + "Here is the answer"(4) = 5 tokens
        """
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="I'll search",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            AIMessage(content="Here is the answer"),
            HumanMessage(content="thanks"),
        ]
        config = CompactionConfig(
            max_recent_messages=100, recent_messages_token_budget=5
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="query"),
        ]
        assert result.to_summarize == [
            AIMessage(
                content="I'll search",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
        ]
        assert result.recent_to_keep == [
            AIMessage(content="Here is the answer"),
            HumanMessage(content="thanks"),
        ]
        mock_estimate.assert_called()

    def test_token_budget_smaller_than_max_recent_messages(self, mock_estimate):
        """Token budget should take precedence over max_recent_messages.

        Token counts:
        - "initial" = 1 token
        - "response one" = 3 tokens
        - "query two" = 2 tokens
        - "response two" = 3 tokens
        - "query three" = 2 tokens
        - "response three" = 3 tokens

        Budget=8: can fit "response three"(3) + "query three"(2) + "response two"(3) = 8 tokens
        """
        messages = [
            HumanMessage(content="initial"),
            AIMessage(content="response one"),
            HumanMessage(content="query two"),
            AIMessage(content="response two"),
            HumanMessage(content="query three"),
            AIMessage(content="response three"),
        ]
        config = CompactionConfig(
            max_recent_messages=10, recent_messages_token_budget=8
        )
        estimator = CompactionTokenEstimator()
        result = slice_for_summarization(messages, config, estimator)

        assert result.leading_context == [
            HumanMessage(content="initial"),
        ]
        assert result.to_summarize == [
            AIMessage(content="response one"),
            HumanMessage(content="query two"),
        ]
        assert result.recent_to_keep == [
            AIMessage(content="response two"),
            HumanMessage(content="query three"),
            AIMessage(content="response three"),
        ]
        mock_estimate.assert_called()


class TestFormatToolCallsAsText:
    """Test suite for _format_tool_calls_as_text function."""

    @pytest.mark.parametrize(
        "tool_calls, expected_fragments",
        [
            pytest.param(
                [{"name": "read_file", "args": {"path": "/foo/bar.py"}}],
                ["[Called tool 'read_file'", '"/foo/bar.py"'],
                id="single_call",
            ),
            pytest.param(
                [
                    {"name": "read_file", "args": {"path": "/foo"}},
                    {
                        "name": "write_file",
                        "args": {"path": "/bar", "content": "hello"},
                    },
                ],
                ["[Called tool 'read_file'", "[Called tool 'write_file'"],
                id="multiple_calls",
            ),
            pytest.param(
                [{"name": "list_files", "args": {}}],
                ["[Called tool 'list_files'", "{}"],
                id="empty_args",
            ),
            pytest.param(
                [{}],
                ["[Called tool 'unknown'"],
                id="missing_fields",
            ),
        ],
    )
    def test_format_tool_calls_as_text(self, tool_calls, expected_fragments):
        result = _format_tool_calls_as_text(tool_calls)
        for fragment in expected_fragments:
            assert fragment in result


class TestStripToolMetadataForLitellm:
    """Test suite for strip_tool_metadata_for_litellm function."""

    @pytest.mark.parametrize(
        "messages, checks",
        [
            pytest.param(
                [
                    AIMessage(
                        content="Let me check.",
                        tool_calls=[
                            {
                                "id": "c1",
                                "name": "read_file",
                                "args": {"path": "/foo"},
                            }
                        ],
                    ),
                    ToolMessage(
                        content="file data",
                        tool_call_id="c1",
                        name="read_file",
                    ),
                    HumanMessage(content="Thanks!"),
                ],
                [
                    lambda r: isinstance(r[0], AIMessage),
                    lambda r: "Let me check." in r[0].content,
                    lambda r: "[Called tool 'read_file'" in r[0].content,
                    lambda r: not r[0].tool_calls,
                    lambda r: isinstance(r[1], HumanMessage),
                    lambda r: "read_file" in r[1].content
                    and "file data" in r[1].content,
                    lambda r: r[2].content == "Thanks!",
                ],
                id="text_plus_tool_calls",
            ),
            pytest.param(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "c1",
                                "name": "search",
                                "args": {"query": "hello"},
                            }
                        ],
                    ),
                ],
                [
                    lambda r: isinstance(r[0], AIMessage),
                    lambda r: "[Called tool 'search'" in r[0].content,
                    lambda r: "hello" in r[0].content,
                    lambda r: not r[0].tool_calls,
                ],
                id="tool_only_no_text",
            ),
            pytest.param(
                [
                    AIMessage(
                        content="Do several things.",
                        tool_calls=[
                            {
                                "id": "c1",
                                "name": "read_file",
                                "args": {"path": "/a"},
                            },
                            {
                                "id": "c2",
                                "name": "write_file",
                                "args": {"path": "/b", "content": "x"},
                            },
                        ],
                    ),
                    ToolMessage(
                        content="contents of a",
                        tool_call_id="c1",
                        name="read_file",
                    ),
                    ToolMessage(
                        content="wrote to b",
                        tool_call_id="c2",
                        name="write_file",
                    ),
                ],
                [
                    lambda r: "[Called tool 'read_file'" in r[0].content,
                    lambda r: "[Called tool 'write_file'" in r[0].content,
                    lambda r: isinstance(r[1], HumanMessage)
                    and "read_file" in r[1].content,
                    lambda r: isinstance(r[2], HumanMessage)
                    and "write_file" in r[2].content,
                ],
                id="multiple_tool_calls_one_message",
            ),
            pytest.param(
                [ToolMessage(content="some result", tool_call_id="c1")],
                [
                    lambda r: isinstance(r[0], HumanMessage),
                    lambda r: "unknown" in r[0].content,
                    lambda r: "some result" in r[0].content,
                ],
                id="tool_message_without_name",
            ),
            pytest.param(
                [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                    SystemMessage(content="Be helpful"),
                ],
                [
                    lambda r: r[0].content == "Hello",
                    lambda r: r[1].content == "Hi there!",
                    lambda r: r[2].content == "Be helpful",
                ],
                id="regular_messages_passthrough",
            ),
            pytest.param(
                [],
                [lambda r: r == []],
                id="empty_messages",
            ),
        ],
    )
    def test_strip_tool_metadata_for_litellm(self, messages, checks):
        result = strip_tool_metadata_for_litellm(messages)
        for check in checks:
            assert check(result)

    def test_strip_tool_metadata_list_content_single_text_plus_tool_use(self):
        """AIMessage with list content [text, tool_use] simplifies to string with tool calls as text."""
        messages = [
            AIMessage(
                content=[
                    {"type": "text", "text": "I'll read the file."},
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "read_file",
                        "input": {"path": "/foo"},
                    },
                ],
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "read_file",
                        "args": {"path": "/foo"},
                    }
                ],
            ),
        ]
        cleaned = strip_tool_metadata_for_litellm(messages)
        content = cleaned[0].content
        # Single text block + tool_use → simplified to string after filtering
        assert isinstance(content, str)
        assert "I'll read the file." in content
        assert "[Called tool 'read_file'" in content
        assert not cleaned[0].tool_calls

    def test_strip_tool_metadata_list_content_multiple_text_plus_tool_use(
        self,
    ):
        """AIMessage with list content [text, text, tool_use] keeps list form."""
        messages = [
            AIMessage(
                content=[
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "read_file",
                        "input": {"path": "/foo"},
                    },
                ],
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "read_file",
                        "args": {"path": "/foo"},
                    }
                ],
            ),
        ]
        cleaned = strip_tool_metadata_for_litellm(messages)
        content = cleaned[0].content
        # Multiple text blocks remain as list
        assert isinstance(content, list)
        text_parts = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert any("First part." in b.get("text", "") for b in text_parts)
        assert any("Second part." in b.get("text", "") for b in text_parts)
        assert any("[Called tool 'read_file'" in b.get("text", "") for b in text_parts)
        assert not any(
            b.get("type") == "tool_use" for b in content if isinstance(b, dict)
        )
        assert not cleaned[0].tool_calls

    def test_strip_tool_metadata_ai_message_without_tool_calls_unchanged(
        self,
    ):
        """AIMessage with no tool_calls is passed through as the same object."""
        msg = AIMessage(content="Just a normal response.")
        cleaned = strip_tool_metadata_for_litellm([msg])
        assert cleaned[0] is msg
