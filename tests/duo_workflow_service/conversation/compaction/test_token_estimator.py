"""Tests for CompactionTokenEstimator class."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.conversation.compaction import CompactionTokenEstimator


class TestCompactionTokenEstimator:
    """Test suite for CompactionTokenEstimator."""

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_arbitrary_messages_with_usage_metadata(self, mock_counter_class):
        """Should use output_tokens from AIMessage usage_metadata."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 0
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_arbitrary_messages(messages)

        assert result == 100
        mock_counter.count_tokens.assert_called_once_with([], include_tool_tokens=False)

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_arbitrary_messages_without_usage_metadata(
        self, mock_counter_class
    ):
        """Should fall back to tiktoken counting."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 50
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        messages = [HumanMessage(content="hello world")]
        result = estimator.estimate_arbitrary_messages(messages)

        assert result == 50
        mock_counter.count_tokens.assert_called_once_with(
            messages, include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_arbitrary_messages_mixed(self, mock_counter_class):
        """Should combine metadata tokens and estimated tokens."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 25
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_arbitrary_messages(messages)

        assert result == 125
        mock_counter.count_tokens.assert_called_once_with(
            [human1, human2], include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_arbitrary_messages_empty(self, mock_counter_class):
        """Should return 0 for empty list."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 0
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        result = estimator.estimate_arbitrary_messages([])

        assert result == 0
        mock_counter.count_tokens.assert_called_once_with([], include_tool_tokens=False)

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_arbitrary_messages_with_tool_messages(self, mock_counter_class):
        """Should handle ToolMessages correctly."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 30
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        human = HumanMessage(content="query")
        ai = AIMessage(
            content="using tool",
            tool_calls=[{"id": "t1", "name": "search", "args": {}}],
        )
        tool = ToolMessage(content="tool result", tool_call_id="t1")
        messages = [human, ai, tool]
        result = estimator.estimate_arbitrary_messages(messages)

        assert result == 30
        mock_counter.count_tokens.assert_called_once_with(
            [human, ai, tool], include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_arbitrary_messages_ai_with_zero_output_tokens(
        self, mock_counter_class
    ):
        """Should fall back to counting when output_tokens is 0."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 20
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        ai = AIMessage(
            content="response",
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 0,
                "total_tokens": 50,
            },
        )
        messages = [ai]
        result = estimator.estimate_arbitrary_messages(messages)

        assert result == 20
        mock_counter.count_tokens.assert_called_once_with(
            [ai], include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_empty_messages(self, mock_counter_class):
        """Should return 0 for empty list."""
        mock_counter = MagicMock()
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        result = estimator.estimate_complete_history([])

        assert result == 0
        mock_counter.count_tokens.assert_not_called()

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_with_total_tokens(self, mock_counter_class):
        """Should use total_tokens as base and count trailing."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 15
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_complete_history(messages)

        assert result == 515
        mock_counter.count_tokens.assert_called_once_with(
            [trailing], include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_ai_at_end(self, mock_counter_class):
        """Should return just total_tokens when latest AI is last message."""
        mock_counter = MagicMock()
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_complete_history(messages)

        assert result == 500
        mock_counter.count_tokens.assert_not_called()

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_no_ai_messages(self, mock_counter_class):
        """Should count all messages when no AIMessage has metadata."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 40
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        messages = [
            HumanMessage(content="query 1"),
            HumanMessage(content="query 2"),
        ]
        result = estimator.estimate_complete_history(messages)

        assert result == 40
        mock_counter.count_tokens.assert_called_once_with(
            messages, include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_ai_without_metadata(self, mock_counter_class):
        """Should count all messages when AIMessage has no usage_metadata."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 35
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
        messages = [
            HumanMessage(content="query"),
            AIMessage(content="response without metadata"),
        ]
        result = estimator.estimate_complete_history(messages)

        assert result == 35
        mock_counter.count_tokens.assert_called_once_with(
            messages, include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_multiple_ai_messages(self, mock_counter_class):
        """Should use the most recent AIMessage with total_tokens."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 10
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_complete_history(messages)

        assert result == 310
        mock_counter.count_tokens.assert_called_once_with(
            [trailing], include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_ai_with_zero_total_tokens(
        self, mock_counter_class
    ):
        """Should skip AIMessage with zero total_tokens and count all."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 45
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_complete_history(messages)

        assert result == 45
        mock_counter.count_tokens.assert_called_once_with(
            messages, include_tool_tokens=False
        )

    @patch(
        "duo_workflow_service.conversation.compaction.token_estimator.TikTokenCounter"
    )
    def test_estimate_complete_history_multiple_trailing_messages(
        self, mock_counter_class
    ):
        """Should count all trailing messages after the latest AI with metadata."""
        mock_counter = MagicMock()
        mock_counter.count_tokens.return_value = 60
        mock_counter_class.return_value = mock_counter

        estimator = CompactionTokenEstimator()
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
        result = estimator.estimate_complete_history(messages)

        assert result == 260
        mock_counter.count_tokens.assert_called_once_with(
            [trailing1, trailing2, trailing3], include_tool_tokens=False
        )
