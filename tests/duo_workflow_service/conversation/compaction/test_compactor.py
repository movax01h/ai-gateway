"""Tests for ConversationCompactor class."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    create_conversation_compactor,
)
from duo_workflow_service.conversation.compaction.compactor import (
    COMPACTION_CONTINUE_MESSAGE,
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
    "duo_workflow_service.conversation.compaction.compactor.get_model_max_context_token_limit"
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
    "duo_workflow_service.conversation.compaction.compactor.get_model_max_context_token_limit"
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

    @patch(
        "duo_workflow_service.conversation.compaction.compactor.get_model_metadata",
        return_value=None,
    )
    def test_is_compaction_call_in_internal_event_extra(
        self, _mock_get_model_metadata, mock_prompt_registry, user
    ):
        """The is_compaction_call flag should be in internal_event_extra."""
        create_conversation_compactor(
            config=CompactionConfig(),
            prompt_registry=mock_prompt_registry,
            user=user,
            agent_name="test_agent",
            workflow_id="test_workflow",
            workflow_type="test_type",
        )

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
    "duo_workflow_service.conversation.compaction.compactor.get_model_max_context_token_limit"
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
    "duo_workflow_service.conversation.compaction.compactor.get_model_max_context_token_limit"
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
