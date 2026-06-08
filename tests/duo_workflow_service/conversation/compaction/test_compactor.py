"""Tests for ConversationCompactor class."""

# pylint: disable=too-many-lines
# Pylint's `too-many-lines` is suppressed here because this file mirrors a
# single source module (compactor.py) and the project enforces 1:1 test/source
# file naming via the file-naming-for-tests lint, so splitting is not an
# option.
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.constants import TAG_NOSTREAM

from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)
from duo_workflow_service.conversation.compaction.compactor import (
    COMPACTION_CONTINUE_MESSAGE,
    COMPACTION_PROMPT_ID,
    COMPACTION_PROMPT_MANUAL_ID,
    CompactionStatus,
    ConversationCompactor,
)

DEFAULT_MAX_RECENT_MESSAGES = 10


def _token_count_side_effect(history_tokens: int, per_turn_tokens: int = 100):
    """Return a TokenEstimator.count side_effect that distinguishes call modes.

    Complete-history calls (``is_complete_history=True``) are used by
    ``should_compact`` / ``original_tokens`` and need to clear the trim
    threshold. Per-turn calls (``is_complete_history=False``) drive
    ``resolve_recent_messages_internal`` and must stay small so recent turns
    are retained.
    """
    return lambda *args, **kwargs: (
        history_tokens if kwargs.get("is_complete_history") else per_turn_tokens
    )


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
)
@patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
class TestConversationCompactorShouldCompact:
    """Test suite for ConversationCompactor.should_compact method."""

    def test_empty_messages_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Empty message list should return False."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = 0

        assert compactor.should_compact([]) is False
        mock_count_tokens.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_below_max_n_recent_messages_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Message count below max_recent_messages should return False."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = 300_000

        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        assert len(messages) < DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is False
        mock_count_tokens.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_exactly_max_n_recent_messages_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Message count exactly at max_recent_messages should return False."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = 300_000

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES)
        ]
        assert len(messages) == DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is False
        mock_count_tokens.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_above_max_n_recent_messages_but_below_trim_threshold_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Message count above max_recent_messages but tokens below threshold should return False."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.6 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]
        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is False
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_exactly_at_trim_threshold_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Token count exactly at trim_threshold should return False (not greater than)."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.7 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        assert compactor.should_compact(messages) is False
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_above_max_n_recent_messages_and_above_trim_threshold_returns_true(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Message count above max_recent_messages and tokens above threshold should return True."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.75 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]
        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is True
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_with_different_model_context_limits(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """should_compact should work with different model context limits."""
        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        small_context = 100_000
        mock_get_max_context.return_value = small_context
        mock_count_tokens.return_value = int(0.75 * small_context)
        assert compactor.should_compact(messages) is True

        large_context = 1_000_000
        mock_get_max_context.return_value = large_context
        mock_count_tokens.return_value = int(0.75 * large_context)
        assert compactor.should_compact(messages) is True

        assert mock_count_tokens.call_count == 2
        assert mock_get_max_context.call_count == 2

    def test_with_high_token_usage_and_few_messages_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """High token usage but few messages should return False."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.9 * max_context)

        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        assert len(messages) < DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is False
        mock_count_tokens.assert_not_called()
        mock_get_max_context.assert_not_called()

    def test_with_many_messages_and_low_token_usage_returns_false(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Many messages but low token usage should return False."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.5 * max_context)

        messages = [
            HumanMessage(content="Short")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 10)
        ]
        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is False
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_with_realistic_conversation_pattern(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Test with realistic conversation pattern (alternating human/AI messages)."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.8 * max_context)

        messages = []
        for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5):
            if i % 2 == 0:
                messages.append(HumanMessage(content=f"Question {i}"))
            else:
                messages.append(AIMessage(content=f"Answer {i}"))

        assert len(messages) > DEFAULT_MAX_RECENT_MESSAGES

        assert compactor.should_compact(messages) is True
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_with_tool_messages_in_conversation(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Test with tool messages in conversation."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.75 * max_context)

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

        assert compactor.should_compact(messages) is True
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_boundary_condition_one_message_above_threshold(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Test boundary: one message above max_recent_messages with tokens just above threshold."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.7 * max_context) + 1

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 1)
        ]

        assert compactor.should_compact(messages) is True
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    def test_with_zero_context_limit(
        self, mock_count_tokens, mock_get_max_context, compactor
    ):
        """Test with zero context limit (edge case)."""
        mock_get_max_context.return_value = 0
        mock_count_tokens.return_value = 100

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        assert compactor.should_compact(messages) is True
        mock_count_tokens.assert_called_once_with(messages, is_complete_history=True)
        mock_get_max_context.assert_called_once()

    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_model_metadata",
        return_value=None,
    )
    def test_with_custom_config_max_recent_messages(
        self,
        _mock_get_model_metadata,
        mock_count_tokens,
        mock_get_max_context,
        mock_prompt_registry,
        user,
    ):
        """Test with custom max_recent_messages in config."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.75 * max_context)

        messages = [HumanMessage(content=f"Message {i}") for i in range(8)]

        compactor = create_conversation_compactor(
            config=CompactionConfig(max_recent_messages=5),
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
        )
        assert compactor.should_compact(messages) is True

    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_model_metadata",
        return_value=None,
    )
    def test_with_custom_config_trim_threshold(
        self,
        _mock_get_model_metadata,
        mock_count_tokens,
        mock_get_max_context,
        mock_prompt_registry,
        user,
    ):
        """Test with custom trim_threshold in config."""
        max_context = 400_000
        mock_get_max_context.return_value = max_context
        mock_count_tokens.return_value = int(0.55 * max_context)

        messages = [
            HumanMessage(content=f"Message {i}")
            for i in range(DEFAULT_MAX_RECENT_MESSAGES + 5)
        ]

        compactor = create_conversation_compactor(
            config=CompactionConfig(trim_threshold=0.5),
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
        )
        assert compactor.should_compact(messages) is True


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
)
class TestConversationCompactorCompact:
    """Test suite for ConversationCompactor.compact method."""

    @pytest.mark.asyncio
    async def test_compact_empty_messages(
        self, _mock_get_max_context, compactor, mock_prompt
    ):
        """Should return unchanged result for empty messages."""
        result = await compactor.compact([])

        assert result.messages == []
        assert result.was_compacted is False
        mock_prompt.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_no_compaction_needed(
        self, mock_get_max_context, compactor, mock_prompt
    ):
        """Should return unchanged when should_compact returns False."""
        mock_get_max_context.return_value = 400_000

        messages = [HumanMessage(content=f"Message {i}") for i in range(5)]
        result = await compactor.compact(messages)

        assert result.messages == messages
        assert result.was_compacted is False
        mock_prompt.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_compact_success(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Should return compacted messages when summarization succeeds."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        summary_message = AIMessage(
            content="Summary of conversation",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_prompt.ainvoke.return_value = summary_message

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
        mock_prompt.ainvoke.assert_called_once()

        # Verify the compaction summary AIMessage exists
        # (second message, after leading HumanMessage context)
        assert isinstance(result.messages[1], AIMessage)
        assert result.messages[1].content == "Summary of conversation"

        # After compaction, last message should be synthetic HumanMessage
        # because the original conversation ends with AIMessage
        assert isinstance(result.messages[-1], HumanMessage)
        assert result.messages[-1].content == COMPACTION_CONTINUE_MESSAGE

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_compact_llm_failure(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Should return original messages and error when LLM fails."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.side_effect = Exception("LLM error")

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
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_model_metadata",
        return_value=None,
    )
    async def test_compact_no_messages_to_summarize(
        self,
        _mock_get_model_metadata,
        mock_count_tokens,
        mock_get_max_context,
        mock_prompt_registry,
        mock_prompt,
        user,
    ):
        """Should return unchanged when slicing leaves nothing to summarize."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        compactor = create_conversation_compactor(
            config=CompactionConfig(
                max_recent_messages=100, recent_messages_token_budget=1_000_000
            ),
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
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
        mock_prompt.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_compact_auto_no_op_when_slicing_yields_empty_to_summarize(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Auto mode logs and returns the original messages when slicing yields no messages to summarize."""
        mock_get_max_context.return_value = 400_000
        # complete-history tokens above threshold -> should_compact True;
        # per-turn cost low so all turns fit in recent_to_keep -> to_summarize empty.
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        # 3 leading HumanMessages + 9 trailing alternating messages.
        # max_recent_messages defaults to 10; remaining (9 msgs) fits entirely.
        messages = [HumanMessage(content=f"L{i}") for i in range(3)]
        for i in range(5):
            messages.append(AIMessage(content=f"A{i}"))
            if i < 4:
                messages.append(HumanMessage(content=f"H{i}"))

        result = await compactor.compact(messages)

        assert result.was_compacted is False
        assert result.messages == messages
        mock_prompt.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_compact_result_has_correct_metadata(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """CompactionResult should have correct token counts and message counts."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_prompt.ainvoke.return_value = summary_message

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
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_compact_does_not_append_human_message_when_last_is_human(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Should not append extra HumanMessage when compacted messages already end with one."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_prompt.ainvoke.return_value = summary_message

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

    @pytest.mark.asyncio
    async def test_invoke_summarizer_strips_tool_metadata(self, compactor, mock_prompt):
        """Tool metadata is unconditionally stripped before invocation."""
        mock_prompt.ainvoke.return_value = AIMessage(content="Summary.")

        await compactor._invoke_summarizer(list(_TOOL_MESSAGES))

        # The prompt.ainvoke receives {"history": messages} where messages
        # have been stripped of tool metadata
        call_kwargs = mock_prompt.ainvoke.call_args[0][0]
        history = call_kwargs["history"]
        for msg in history:
            if isinstance(msg, AIMessage):
                assert not msg.tool_calls
            assert not isinstance(msg, ToolMessage)

        # Verify tool call context is preserved as human-readable text
        all_content = " ".join(
            m.content if isinstance(m.content, str) else str(m.content) for m in history
        )
        assert "[Called tool 'read_file'" in all_content
        assert "[Tool result for 'read_file']" in all_content


class TestIsCompactionCallFlag:
    """Test that is_compaction_call is set in internal_event_extra."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_model_metadata",
        return_value=None,
    )
    async def test_is_compaction_call_in_internal_event_extra(
        self, _mock_get_model_metadata, mock_prompt_registry, mock_prompt, user
    ):
        """The is_compaction_call flag should be in internal_event_extra."""
        compactor = create_conversation_compactor(
            config=CompactionConfig(),
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
        )

        mock_prompt.ainvoke.return_value = AIMessage(content="summary")
        await compactor.compact([HumanMessage(content="x")], is_manual=True)

        call_kwargs = mock_prompt_registry.get_on_behalf.call_args
        extra = call_kwargs.kwargs.get(
            "internal_event_extra", call_kwargs[1].get("internal_event_extra")
        )
        assert extra["is_compaction_call"] is True
        assert extra["agent_name"] == "test_agent"
        assert extra["workflow_id"] == "test_workflow"
        assert extra["workflow_type"] == "test_type"
        assert "operation_type" not in extra


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
)
@patch("duo_workflow_service.conversation.compaction.compactor.duo_workflow_metrics")
class TestCompactorPrometheusMetrics:
    """Test Prometheus metric recording during compaction."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_success_increments_counter_and_records_duration(
        self,
        mock_count_tokens,
        mock_metrics,
        mock_get_max_context,
        compactor,
        mock_prompt,
    ):
        """Successful compaction should increment counter and observe duration."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_prompt.ainvoke.return_value = summary_message

        messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
        for i in range(15):
            messages.append(HumanMessage(content=f"M{i}"))
            messages.append(AIMessage(content=f"R{i}"))

        await compactor.compact(messages)

        mock_metrics.count_compaction_execution.assert_called_once_with(
            flow_type="test_type",
            agent_name="test_agent",
            status=CompactionStatus.SUCCESS,
        )
        mock_metrics.time_compaction_llm.assert_called_once_with(
            flow_type="test_type",
            agent_name="test_agent",
        )

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_error_increments_counter(
        self,
        mock_count_tokens,
        mock_metrics,
        mock_get_max_context,
        compactor,
        mock_prompt,
    ):
        """Failed compaction should increment counter with error status."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        mock_prompt.ainvoke.side_effect = Exception("LLM error")

        messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
        for i in range(15):
            messages.append(HumanMessage(content=f"M{i}"))
            messages.append(AIMessage(content=f"R{i}"))

        await compactor.compact(messages)

        mock_metrics.count_compaction_execution.assert_called_once_with(
            flow_type="test_type",
            agent_name="test_agent",
            status=CompactionStatus.ERROR,
        )


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
)
class TestCompactorSnowplowEvents:
    """Test Snowplow event firing during compaction."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.duo_workflow_metrics"
    )
    async def test_fires_compaction_event_on_success(
        self,
        _mock_metrics,
        mock_count_tokens,
        mock_get_max_context,
        compactor_with_events,
        mock_prompt,
        mock_internal_events_client,
    ):
        """Should fire compaction_executed event with correct fields on success."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_prompt.ainvoke.return_value = summary_message

        messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
        for i in range(15):
            messages.append(HumanMessage(content=f"M{i}"))
            messages.append(AIMessage(content=f"R{i}"))

        await compactor_with_events.compact(messages)

        mock_internal_events_client.track_event.assert_called_once()
        call_kwargs = mock_internal_events_client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == "duo_workflow_compaction_executed"
        assert call_kwargs["category"] == ConversationCompactor.__name__
        additional_props = call_kwargs["additional_properties"]
        assert additional_props.label == "test_agent"
        assert additional_props.property == "workflow_id"
        assert additional_props.value == "test_workflow"
        assert additional_props.extra["status"] == "success"
        assert additional_props.extra["model_name"] == "unknown"
        assert additional_props.extra["operation_type"] == "compaction_auto"
        assert additional_props.extra["compaction_input_tokens"] == 1000
        assert additional_props.extra["compaction_output_tokens"] == 200
        assert additional_props.extra["tokens_before"] > 0
        assert "messages_summarized" in additional_props.extra
        assert "token_budget" in additional_props.extra
        assert "max_context_tokens" in additional_props.extra
        assert "duration_seconds" in additional_props.extra

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.duo_workflow_metrics"
    )
    async def test_fires_compaction_event_on_error(
        self,
        _mock_metrics,
        mock_count_tokens,
        mock_get_max_context,
        compactor_with_events,
        mock_prompt,
        mock_internal_events_client,
    ):
        """Should fire compaction_executed event with error_type on failure."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        mock_prompt.ainvoke.side_effect = RuntimeError("LLM error")

        messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
        for i in range(15):
            messages.append(HumanMessage(content=f"M{i}"))
            messages.append(AIMessage(content=f"R{i}"))

        await compactor_with_events.compact(messages)

        mock_internal_events_client.track_event.assert_called_once()
        call_kwargs = mock_internal_events_client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == "duo_workflow_compaction_executed"
        assert call_kwargs["category"] == ConversationCompactor.__name__
        additional_props = call_kwargs["additional_properties"]
        assert additional_props.extra["status"] == "error"
        assert additional_props.extra["model_name"] == "unknown"
        assert additional_props.extra["operation_type"] == "compaction_auto"
        assert additional_props.extra["error_type"] == "RuntimeError"
        assert "duration_seconds" in additional_props.extra

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.duo_workflow_metrics"
    )
    async def test_fire_compaction_event_noop_without_client(
        self,
        _mock_metrics,
        mock_count_tokens,
        mock_get_max_context,
        compactor,
        mock_prompt,
    ):
        """_fire_compaction_event should be a no-op when internal_events_client is None."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        summary_message = AIMessage(
            content="Summary",
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 200,
                "total_tokens": 1200,
            },
        )
        mock_prompt.ainvoke.return_value = summary_message

        messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
        for i in range(15):
            messages.append(HumanMessage(content=f"M{i}"))
            messages.append(AIMessage(content=f"R{i}"))

        # compactor fixture has no internal_events_client (None)
        result = await compactor.compact(messages)

        # Should succeed without error -- no event is fired
        assert result.was_compacted is True

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.duo_workflow_metrics"
    )
    async def test_fire_compaction_event_noop_when_prompt_load_fails(
        self,
        _mock_metrics,
        mock_count_tokens,
        mock_get_max_context,
        compactor_with_events,
        mock_prompt_registry,
        mock_internal_events_client,
    ):
        """_fire_compaction_event should be a no-op when the prompt failed to load.

        When prompt_registry.get_on_behalf raises, the per-mode prompt cache stays empty; firing the event would require
        re-attempting the load (which would just raise again), so the event is silently dropped.
        """
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.return_value = int(0.8 * 400_000)

        mock_prompt_registry.get_on_behalf.side_effect = RuntimeError(
            "prompt registry failure"
        )

        messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
        for i in range(15):
            messages.append(HumanMessage(content=f"M{i}"))
            messages.append(AIMessage(content=f"R{i}"))

        result = await compactor_with_events.compact(messages)

        assert result.was_compacted is False
        assert isinstance(result.error, RuntimeError)
        mock_internal_events_client.track_event.assert_not_called()


def _summary_message(content: str = "Summary", message_id: str | None = None):
    usage_metadata = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "total_tokens": 1200,
    }
    if message_id is None:
        return AIMessage(content=content, usage_metadata=usage_metadata)
    return AIMessage(content=content, usage_metadata=usage_metadata, id=message_id)


def _large_history():
    messages = [HumanMessage(content="Initial query"), AIMessage(content="R1")]
    for i in range(15):
        messages.append(HumanMessage(content=f"M{i}"))
        messages.append(AIMessage(content=f"R{i}"))
    return messages


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
)
class TestCompactManualMode:
    """Tests for manual-mode behaviors of ConversationCompactor.compact."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_bypasses_should_compact(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Manual mode should compact even when below thresholds."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.1 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message(message_id="summary-1")

        short_history = [HumanMessage(content=f"M{i}") for i in range(3)] + [
            AIMessage(content="R0"),
            HumanMessage(content="follow up"),
            AIMessage(content="R1"),
        ]

        result = await compactor.compact(short_history, is_manual=True)

        assert result.was_compacted is True
        assert result.summary is not None
        assert result.summary.id == "summary-1"
        mock_prompt.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_propagates_user_instruction(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """user_instruction should be forwarded into prompt inputs."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(
            _large_history(),
            is_manual=True,
            user_instruction="focus on auth bug",
        )

        call_args, _ = mock_prompt.ainvoke.call_args
        inputs = call_args[0]
        assert inputs.get("user_instruction") == "focus on auth bug"

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_omits_user_instruction_when_none(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """user_instruction key must be absent when not provided."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(_large_history(), is_manual=True)

        call_args, _ = mock_prompt.ainvoke.call_args
        inputs = call_args[0]
        assert "user_instruction" not in inputs

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_skips_synthetic_human_message(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Manual mode must never append the synthetic COMPACTION_CONTINUE_MESSAGE."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message()

        result = await compactor.compact(_large_history(), is_manual=True)

        assert result.was_compacted is True
        contents = [m.content for m in result.messages if isinstance(m, HumanMessage)]
        assert COMPACTION_CONTINUE_MESSAGE not in contents

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_streams_summary_no_nostream_tag(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Manual mode should not apply the TAG_NOSTREAM tag."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(_large_history(), is_manual=True)

        _, call_kwargs = mock_prompt.ainvoke.call_args
        tags = call_kwargs["config"]["tags"]
        assert TAG_NOSTREAM not in tags

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_auto_uses_nostream_tag(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Auto mode must keep TAG_NOSTREAM."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(_large_history(), is_manual=False)

        _, call_kwargs = mock_prompt.ainvoke.call_args
        tags = call_kwargs["config"]["tags"]
        assert TAG_NOSTREAM in tags

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_user_instruction_included_in_prompt_overhead(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Prompt overhead must reflect the rendered template, including any user instruction, so post-compaction token
        accounting stays accurate."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(
            _large_history(),
            is_manual=True,
            user_instruction="focus on auth",
        )

        # _calculate_compacted_tokens renders the template via format_messages
        # to size the prompt overhead. That render must carry the user_instruction
        # so the overhead estimate matches what _invoke_summarizer actually sent.
        format_calls = mock_prompt.prompt_tpl.format_messages.call_args_list
        assert format_calls, "expected at least one format_messages call for overhead"
        assert any(
            call.kwargs.get("user_instruction") == "focus on auth"
            for call in format_calls
        )

    @pytest.mark.asyncio
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_force_summarizes_all_when_to_summarize_empty(
        self, mock_count_tokens, mock_get_max_context, compactor, mock_prompt
    ):
        """Manual mode summarizes the entire history when normal slicing is empty."""
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        mock_prompt.ainvoke.return_value = _summary_message(message_id="summary-all")

        history = [HumanMessage(content=f"msg {i}") for i in range(3)]

        result = await compactor.compact(history, is_manual=True)

        assert result.was_compacted is True
        call_args, _ = mock_prompt.ainvoke.call_args
        passed_history = call_args[0]["history"]
        assert passed_history == history
        assert result.messages == [result.summary]


@patch(
    "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
)
class TestCompactorOperationType:
    """Tests for the operation_type carried in the Snowplow event per mode."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "is_manual, expected_prompt_id, expected_operation_type",
        [
            (False, COMPACTION_PROMPT_ID, "compaction_auto"),
            (True, COMPACTION_PROMPT_MANUAL_ID, "compaction_manual"),
        ],
        ids=["auto", "manual"],
    )
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.duo_workflow_metrics"
    )
    async def test_compaction_event_uses_correct_operation_type(
        self,
        _mock_metrics,
        mock_count_tokens,
        mock_get_max_context,
        is_manual,
        expected_prompt_id,
        expected_operation_type,
        compactor_with_events,
        mock_prompt_registry,
        mock_prompt,
        mock_prompt_manual,
        mock_internal_events_client,
    ):
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))

        def fake_get_on_behalf(_user, prompt_id, *_args, **_kwargs):
            if prompt_id == COMPACTION_PROMPT_MANUAL_ID:
                return mock_prompt_manual
            return mock_prompt

        mock_prompt_registry.get_on_behalf.side_effect = fake_get_on_behalf
        mock_prompt.ainvoke.return_value = _summary_message()
        mock_prompt_manual.ainvoke.return_value = _summary_message()

        await compactor_with_events.compact(_large_history(), is_manual=is_manual)

        event_call = mock_internal_events_client.track_event.call_args
        additional_props = event_call.kwargs["additional_properties"]
        assert additional_props.extra["operation_type"] == expected_operation_type

        called_ids = [
            c.args[1] for c in mock_prompt_registry.get_on_behalf.call_args_list
        ]
        assert expected_prompt_id in called_ids


class TestCompactorLazyPromptLoad:
    """Tests verifying prompts are only loaded on first compact() call per mode."""

    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_model_metadata",
        return_value=None,
    )
    def test_init_does_not_load_any_prompt(
        self, _mock_get_model_metadata, mock_prompt_registry, user
    ):
        create_conversation_compactor(
            config=CompactionConfig(),
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
        )
        mock_prompt_registry.get_on_behalf.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
    )
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_auto_compact_loads_only_auto_prompt(
        self,
        mock_count_tokens,
        mock_get_max_context,
        compactor,
        mock_prompt_registry,
        mock_prompt,
    ):
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))
        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(_large_history(), is_manual=False)

        prompt_ids_called = [
            c.args[1] for c in mock_prompt_registry.get_on_behalf.call_args_list
        ]
        assert COMPACTION_PROMPT_ID in prompt_ids_called
        assert COMPACTION_PROMPT_MANUAL_ID not in prompt_ids_called

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_current_model_max_context_token_limit"
    )
    @patch("duo_workflow_service.conversation.token_estimator.TokenEstimator.count")
    async def test_manual_compact_loads_only_manual_prompt(
        self,
        mock_count_tokens,
        mock_get_max_context,
        compactor,
        mock_prompt_registry,
        mock_prompt,
    ):
        mock_get_max_context.return_value = 400_000
        mock_count_tokens.side_effect = _token_count_side_effect(int(0.8 * 400_000))
        mock_prompt.ainvoke.return_value = _summary_message()

        await compactor.compact(_large_history(), is_manual=True)

        prompt_ids_called = [
            c.args[1] for c in mock_prompt_registry.get_on_behalf.call_args_list
        ]
        assert COMPACTION_PROMPT_MANUAL_ID in prompt_ids_called
        assert COMPACTION_PROMPT_ID not in prompt_ids_called
