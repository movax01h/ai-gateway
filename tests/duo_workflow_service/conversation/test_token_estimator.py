"""Tests for conversation token estimator module."""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.conversation.token_estimator import TokenEstimator

PATCH_COUNT_APPROX = (
    "duo_workflow_service.conversation.token_estimator.count_tokens_approximately"
)


def count_tokens(messages, *, is_complete_history):
    """Test helper that delegates to a fresh TokenEstimator instance."""
    return TokenEstimator().count(messages, is_complete_history=is_complete_history)


class TestCountTokensArbitrary:
    """Tests for count_tokens with is_complete_history=False (arbitrary messages)."""

    @patch(PATCH_COUNT_APPROX)
    def test_with_usage_metadata(self, mock_count_approx):
        """Should use output_tokens from AIMessage usage_metadata."""
        mock_count_approx.return_value = 0

        messages = [
            AIMessage(
                content="response",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "total_tokens": 150,
                },
            ),
        ]
        result = count_tokens(messages, is_complete_history=False)

        assert result == 100
        mock_count_approx.assert_called_once_with(messages=[])

    @patch(PATCH_COUNT_APPROX)
    def test_without_usage_metadata(self, mock_count_approx):
        """Should fall back to approximate counting."""
        mock_count_approx.return_value = 50

        messages = [HumanMessage(content="hello world")]
        result = count_tokens(messages, is_complete_history=False)

        assert result == 50
        mock_count_approx.assert_called_once_with(messages=messages)

    @patch(PATCH_COUNT_APPROX)
    def test_mixed(self, mock_count_approx):
        """Should combine metadata tokens and estimated tokens."""
        mock_count_approx.return_value = 25

        human1 = HumanMessage(content="hello")
        human2 = HumanMessage(content="follow up")
        messages = [
            human1,
            AIMessage(
                content="response",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 100,
                    "total_tokens": 150,
                },
            ),
            human2,
        ]
        result = count_tokens(messages, is_complete_history=False)

        assert result == 125
        mock_count_approx.assert_called_once_with(messages=[human1, human2])

    @patch(PATCH_COUNT_APPROX)
    def test_empty(self, mock_count_approx):
        """Should return 0 for empty list."""
        mock_count_approx.return_value = 0

        result = count_tokens([], is_complete_history=False)

        assert result == 0
        mock_count_approx.assert_called_once_with(messages=[])

    @patch(PATCH_COUNT_APPROX)
    def test_with_tool_messages(self, mock_count_approx):
        """Should handle ToolMessages correctly."""
        mock_count_approx.return_value = 30

        human = HumanMessage(content="query")
        ai = AIMessage(
            content="using tool",
            tool_calls=[{"id": "t1", "name": "search", "args": {}}],
        )
        tool = ToolMessage(content="tool result", tool_call_id="t1")
        messages = [human, ai, tool]
        result = count_tokens(messages, is_complete_history=False)

        assert result == 30
        mock_count_approx.assert_called_once_with(messages=[human, ai, tool])

    @patch(PATCH_COUNT_APPROX)
    def test_ai_with_zero_output_tokens(self, mock_count_approx):
        """Should fall back to counting when output_tokens is 0."""
        mock_count_approx.return_value = 20

        ai = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 0,
                "total_tokens": 50,
            },
        )
        messages = [ai]
        result = count_tokens(messages, is_complete_history=False)

        assert result == 20
        mock_count_approx.assert_called_once_with(messages=[ai])


class TestCountTokensHistory:
    """Tests for count_tokens with is_complete_history=True (complete history)."""

    @patch(PATCH_COUNT_APPROX)
    def test_empty(self, mock_count_approx):
        """Should return 0 for empty list."""
        result = count_tokens([], is_complete_history=True)

        assert result == 0
        mock_count_approx.assert_not_called()

    @patch(PATCH_COUNT_APPROX)
    def test_with_total_tokens(self, mock_count_approx):
        """Should use total_tokens as base and count trailing."""
        mock_count_approx.return_value = 15

        trailing = HumanMessage(content="follow up")
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="response",
                usage_metadata={
                    "input_tokens": 200,
                    "output_tokens": 300,
                    "total_tokens": 500,
                },
            ),
            trailing,
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 515
        mock_count_approx.assert_called_once_with(messages=[trailing])

    @patch(PATCH_COUNT_APPROX)
    def test_ai_at_end(self, mock_count_approx):
        """Should return just total_tokens when latest AI is last message."""
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="response",
                usage_metadata={
                    "input_tokens": 200,
                    "output_tokens": 300,
                    "total_tokens": 500,
                },
            ),
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 500
        mock_count_approx.assert_not_called()

    @patch(PATCH_COUNT_APPROX)
    def test_no_ai_messages(self, mock_count_approx):
        """Should count all messages when no AIMessage has metadata."""
        mock_count_approx.return_value = 40

        messages = [
            HumanMessage(content="query 1"),
            HumanMessage(content="query 2"),
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 40
        mock_count_approx.assert_called_once_with(messages=messages)

    @patch(PATCH_COUNT_APPROX)
    def test_ai_without_metadata(self, mock_count_approx):
        """Should count all messages when AIMessage has no usage_metadata."""
        mock_count_approx.return_value = 35

        messages = [
            HumanMessage(content="query"),
            AIMessage(content="response without metadata"),
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 35
        mock_count_approx.assert_called_once_with(messages=messages)

    @patch(PATCH_COUNT_APPROX)
    def test_multiple_ai_messages(self, mock_count_approx):
        """Should use the most recent AIMessage with total_tokens."""
        mock_count_approx.return_value = 10

        trailing = HumanMessage(content="query 3")
        messages = [
            HumanMessage(content="query 1"),
            AIMessage(
                content="response 1",
                usage_metadata={
                    "input_tokens": 50,
                    "output_tokens": 50,
                    "total_tokens": 100,
                },
            ),
            HumanMessage(content="query 2"),
            AIMessage(
                content="response 2",
                usage_metadata={
                    "input_tokens": 150,
                    "output_tokens": 150,
                    "total_tokens": 300,
                },
            ),
            trailing,
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 310
        mock_count_approx.assert_called_once_with(messages=[trailing])

    @patch(PATCH_COUNT_APPROX)
    def test_ai_with_zero_total_tokens(self, mock_count_approx):
        """Should skip AIMessage with zero total_tokens and count all."""
        mock_count_approx.return_value = 45

        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="response",
                usage_metadata={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            ),
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 45
        mock_count_approx.assert_called_once_with(messages=messages)

    @patch(PATCH_COUNT_APPROX)
    def test_multiple_trailing_messages(self, mock_count_approx):
        """Should count all trailing messages after the latest AI with metadata."""
        mock_count_approx.return_value = 60

        trailing1 = HumanMessage(content="follow up 1")
        trailing2 = AIMessage(content="response without metadata")
        trailing3 = HumanMessage(content="follow up 2")
        messages = [
            HumanMessage(content="query"),
            AIMessage(
                content="response",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 100,
                    "total_tokens": 200,
                },
            ),
            trailing1,
            trailing2,
            trailing3,
        ]
        result = count_tokens(messages, is_complete_history=True)

        assert result == 260
        mock_count_approx.assert_called_once_with(
            messages=[trailing1, trailing2, trailing3]
        )
