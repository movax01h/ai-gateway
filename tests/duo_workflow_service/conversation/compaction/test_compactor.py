"""Tests for ConversationCompactor class."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM
from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    CompactionTokenEstimator,
    create_conversation_compactor,
)
from duo_workflow_service.conversation.compaction.compactor import (
    COMPACTION_CONTINUE_MESSAGE,
)

DEFAULT_MAX_RECENT_MESSAGES = 10


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_model_max_context_token_limit"
)
@patch.object(CompactionTokenEstimator, "estimate_complete_history")
class TestConversationCompactorShouldCompact:
    """Test suite for ConversationCompactor.should_compact method."""

    def _create_compactor(self, config=None):
        """Helper to create compactor with mock LLM."""
        mock_llm = Mock()
        return create_conversation_compactor(
            llm_model=mock_llm,
            config=config or CompactionConfig(),
        )

    def test_empty_messages_returns_false(self, mock_estimate, mock_get_max_context):
        """Empty message list should return False."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = 0

        compactor = self._create_compactor()
        assert compactor.should_compact([]) is False
        mock_estimate.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_below_max_n_recent_messages_returns_false(
        self, mock_estimate, mock_get_max_context
    ):
        """Message count below max_recent_messages should return False."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = 300_000

        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        assert len(messages) < DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is False
        mock_estimate.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_exactly_max_n_recent_messages_returns_false(
        self, mock_estimate, mock_get_max_context
    ):
        """Message count exactly at max_recent_messages should return False."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = 300_000

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES)
        ]
        assert len(messages) == DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is False
        mock_estimate.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_above_max_n_recent_messages_but_below_trim_threshold_returns_false(
        self, mock_estimate, mock_get_max_context
    ):
        """Message count above max_recent_messages but tokens below threshold should return False."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.6 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]
        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is False
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_exactly_at_trim_threshold_returns_false(
        self, mock_estimate, mock_get_max_context
    ):
        """Token count exactly at trim_threshold should return False (not greater than)."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.7 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is False
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_above_max_n_recent_messages_and_above_trim_threshold_returns_true(
        self, mock_estimate, mock_get_max_context
    ):
        """Message count above max_recent_messages and tokens above threshold should return True."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.75 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]
        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is True
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_with_different_model_context_limits(
        self, mock_estimate, mock_get_max_context
    ):
        """should_compact should work with different model context limits."""
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        compactor = self._create_compactor()

        small_context = 100_000
        mock_get_max_context.return_value = small_context
        mock_estimate.return_value = int(0.75 * small_context)
        assert compactor.should_compact(messages) is True

        large_context = 1_000_000
        mock_get_max_context.return_value = large_context
        mock_estimate.return_value = int(0.75 * large_context)
        assert compactor.should_compact(messages) is True

        assert mock_estimate.call_count == 2
        assert mock_get_max_context.call_count == 2

    def test_with_high_token_usage_and_few_messages_returns_false(
        self, mock_estimate, mock_get_max_context
    ):
        """High token usage but few messages should return False."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.9 * max_context)

        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        assert len(messages) < DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is False
        mock_estimate.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_with_many_messages_and_low_token_usage_returns_false(
        self, mock_estimate, mock_get_max_context
    ):
        """Many messages but low token usage should return False."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.5 * max_context)

        messages = [
            HumanMessage(content="Short")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 10)
        ]
        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is False
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_with_realistic_conversation_pattern(
        self, mock_estimate, mock_get_max_context
    ):
        """Test with realistic conversation pattern (alternating human/AI messages)."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.8 * max_context)

        messages = []
        for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"Question {i}"))
            else:
                messages.append(AIMessage(content=f"Answer {i}"))

        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is True
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_with_tool_messages_in_conversation(
        self, mock_estimate, mock_get_max_context
    ):
        """Test with tool messages in conversation."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.75 * max_context)

        messages = [
            HumanMessage(content="Help me"),
            AIMessage(
                content="I'll search",
                tool_calls=[{"id": "tool_1", "name": "search", "args": {}}],
            ),
            ToolMessage(content="Search result", tool_call_id="tool_1"),
            AIMessage(content="Found it"),
        ]
        for i in range(DEFAULT_MAX_RECENT_MESSAGES + 2):
            messages.append(HumanMessage(content=f"Message {i}"))

        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is True
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_boundary_condition_one_message_above_threshold(
        self, mock_estimate, mock_get_max_context
    ):
        """Test boundary: one message above max_recent_messages with tokens just above threshold."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.7 * max_context) + 1

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 1)
        ]

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is True
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_with_zero_context_limit(self, mock_estimate, mock_get_max_context):
        """Test with zero context limit (edge case)."""
        mock_get_max_context.return_value = 0
        mock_estimate.return_value = 100

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        compactor = self._create_compactor()
        assert compactor.should_compact(messages) is True
        mock_estimate.assert_called_once_with(messages)
        mock_get_max_context.assert_called_once()

    def test_with_custom_config_max_recent_messages(
        self, mock_estimate, mock_get_max_context
    ):
        """Test with custom max_recent_messages in config."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.75 * max_context)

        messages = [HumanMessage(content=f"Message {i}") for i in range(8)]

        config = CompactionConfig(max_recent_messages=5)
        compactor = self._create_compactor(config=config)
        assert compactor.should_compact(messages) is True

    def test_with_custom_config_trim_threshold(
        self, mock_estimate, mock_get_max_context
    ):
        """Test with custom trim_threshold in config."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_estimate.return_value = int(0.55 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        config = CompactionConfig(trim_threshold=0.5)
        compactor = self._create_compactor(config=config)
        assert compactor.should_compact(messages) is True


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_model_max_context_token_limit"
)
class TestConversationCompactorCompact:
    """Test suite for ConversationCompactor.compact method."""

    @pytest.mark.asyncio
    async def test_compact_empty_messages(self, mock_get_max_context):
        """Should return unchanged result for empty messages."""
        mock_llm = AsyncMock()
        compactor = create_conversation_compactor(
            llm_model=mock_llm, config=CompactionConfig()
        )

        result = await compactor.compact([])

        assert result.messages == []
        assert result.was_compacted is False
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_no_compaction_needed(self, mock_get_max_context):
        """Should return unchanged when should_compact returns False."""
        mock_get_max_context.return_value = 400_000
        mock_llm = AsyncMock()
        compactor = create_conversation_compactor(
            llm_model=mock_llm, config=CompactionConfig()
        )

        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        result = await compactor.compact(messages)

        assert result.messages == messages
        assert result.was_compacted is False
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(CompactionTokenEstimator, "estimate_complete_history")
    async def test_compact_success(self, mock_estimate, mock_get_max_context):
        """Should return compacted messages when summarization succeeds."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary of conversation",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = summary_message

        compactor = create_conversation_compactor(
            llm_model=mock_llm, config=CompactionConfig()
        )

        messages = [
            HumanMessage(content="Initial query"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Follow up"),
            AIMessage(content="Response 2"),
        ]
        for i in range(15):
            messages.append(HumanMessage(content=f"Message {i}"))
            messages.append(AIMessage(content=f"Response {i}"))

        result = await compactor.compact(messages)

        assert result.was_compacted is True
        assert len(result.messages) < len(messages)
        mock_llm.ainvoke.assert_called_once()

        # Verify the compaction summary AIMessage exists
        # (second message, after leading HumanMessage context)
        assert isinstance(result.messages[1], AIMessage)
        assert result.messages[1].content == "Summary of conversation"

        # After compaction, last message should be synthetic HumanMessage
        # because the original conversation ends with AIMessage
        assert isinstance(result.messages[-1], HumanMessage)
        assert result.messages[-1].content == COMPACTION_CONTINUE_MESSAGE

    @pytest.mark.asyncio
    @patch.object(CompactionTokenEstimator, "estimate_complete_history")
    async def test_compact_llm_failure(self, mock_estimate, mock_get_max_context):
        """Should return original messages and error when LLM fails."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = int(0.8 * 400_000)

        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM error")

        compactor = create_conversation_compactor(
            llm_model=mock_llm, config=CompactionConfig()
        )

        messages = [
            HumanMessage(content="Initial query"),
            AIMessage(content="Response 1"),
        ]
        for i in range(15):
            messages.append(HumanMessage(content=f"Message {i}"))
            messages.append(AIMessage(content=f"Response {i}"))

        result = await compactor.compact(messages)

        assert result.was_compacted is False
        assert result.messages == messages
        assert result.error is not None

    @pytest.mark.asyncio
    @patch.object(CompactionTokenEstimator, "estimate_complete_history")
    @patch.object(
        CompactionTokenEstimator, "estimate_arbitrary_messages", return_value=50000
    )
    async def test_compact_no_messages_to_summarize(
        self, mock_estimate_arbitrary, mock_estimate_complete, mock_get_max_context
    ):
        """Should return unchanged when slicing leaves nothing to summarize."""
        mock_get_max_context.return_value = 400_000
        mock_estimate_complete.return_value = int(0.8 * 400_000)

        mock_llm = AsyncMock()
        compactor = create_conversation_compactor(
            llm_model=mock_llm,
            config=CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=1_000_000
            ),
        )

        messages = [
            HumanMessage(content="Query"),
            AIMessage(content="Response"),
        ]
        for i in range(15):
            messages.append(HumanMessage(content=f"Message {i}"))
            messages.append(AIMessage(content=f"Response {i}"))

        result = await compactor.compact(messages)

        assert result.was_compacted is False
        mock_llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(CompactionTokenEstimator, "estimate_complete_history")
    async def test_compact_result_has_correct_metadata(
        self, mock_estimate, mock_get_max_context
    ):
        """CompactionResult should have correct token counts and message counts."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = summary_message

        compactor = create_conversation_compactor(
            llm_model=mock_llm, config=CompactionConfig()
        )

        messages = [
            HumanMessage(content="Initial"),
            AIMessage(content="Response 1"),
        ]
        for i in range(15):
            messages.append(HumanMessage(content=f"Message {i}"))
            messages.append(AIMessage(content=f"Response {i}"))

        result = await compactor.compact(messages)

        assert result.was_compacted is True
        assert result.tokens_before > 0
        assert result.messages_summarized > 0

    @pytest.mark.asyncio
    @patch.object(CompactionTokenEstimator, "estimate_complete_history")
    async def test_compact_does_not_append_human_message_when_last_is_human(
        self, mock_estimate, mock_get_max_context
    ):
        """Should not append extra HumanMessage when compacted messages already end with one."""
        mock_get_max_context.return_value = 400_000
        mock_estimate.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = summary_message

        compactor = create_conversation_compactor(
            llm_model=mock_llm, config=CompactionConfig()
        )

        # Conversation ending with HumanMessage
        messages = [
            HumanMessage(content="Initial query"),
            AIMessage(content="Response 1"),
        ]
        for i in range(15):
            messages.append(HumanMessage(content=f"Follow-up {i}"))
            messages.append(AIMessage(content=f"Response {i}"))
        messages.append(HumanMessage(content="Final question"))

        result = await compactor.compact(messages)

        assert result.was_compacted is True

        # Verify the compaction summary AIMessage exists
        # (second message, after leading HumanMessage context)
        assert isinstance(result.messages[1], AIMessage)
        assert result.messages[1].content == "Summary"

        # Last message should be the user's actual message, not the synthetic one
        assert isinstance(result.messages[-1], HumanMessage)
        assert result.messages[-1].content != COMPACTION_CONTINUE_MESSAGE


_TOOL_MESSAGES = [
    AIMessage(
        content="I'll check.",
        tool_calls=[{"id": "c1", "name": "read_file", "args": {"path": "/foo"}}],
    ),
    ToolMessage(content="file data", tool_call_id="c1", name="read_file"),
]


class TestInvokeSummarizerToolMetadataStripping:
    """Tests for _invoke_summarizer tool metadata handling."""

    def _create_compactor(self, mock_llm):
        """Helper to create compactor with the given mock LLM."""
        return create_conversation_compactor(
            llm_model=mock_llm,
            config=CompactionConfig(),
        )

    @pytest.mark.asyncio
    async def test_invoke_summarizer_strips_tool_metadata_for_litellm(self):
        """When self._llm is ChatLiteLLM, tool metadata is stripped before invocation."""
        mock_llm = MagicMock(spec=ChatLiteLLM)
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Summary."))
        compactor = self._create_compactor(mock_llm)

        await compactor._invoke_summarizer(list(_TOOL_MESSAGES))

        call_args = mock_llm.ainvoke.call_args[0][0]
        for msg in call_args:
            if isinstance(msg, AIMessage):
                assert not msg.tool_calls
            assert not isinstance(msg, ToolMessage)

        # Verify tool call context is preserved as human-readable text
        inner_msgs = call_args[1:-1]  # strip system/user prompt wrappers
        all_content = " ".join(
            m.content if isinstance(m.content, str) else str(m.content)
            for m in inner_msgs
        )
        assert "[Called tool 'read_file'" in all_content
        assert "[Tool result for 'read_file']" in all_content

    @pytest.mark.asyncio
    async def test_invoke_summarizer_does_not_strip_for_non_litellm(self):
        """When self._llm is NOT ChatLiteLLM, messages are passed through unchanged."""
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Summary."))
        compactor = self._create_compactor(mock_llm)

        await compactor._invoke_summarizer(list(_TOOL_MESSAGES))

        call_args = mock_llm.ainvoke.call_args[0][0]
        # The original tool-bearing messages should be present (sandwiched
        # between the system and user prompts)
        inner_msgs = call_args[1:-1]  # strip system/user prompt wrappers
        assert any(isinstance(m, AIMessage) and m.tool_calls for m in inner_msgs)
        assert any(isinstance(m, ToolMessage) for m in inner_msgs)
