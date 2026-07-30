"""Tests for the ``maybe_compact_history`` compatibility shim.

After the history-optimizer refactor (MR 1), ``maybe_compact_history``
builds a transient ``HistoryOptimizerPipeline`` and returns the same
``(messages, CompactionResult | None)`` tuple shape as before. These tests
verify the shim continues to honor the legacy contract end-to-end.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.conversation.compaction import CompactionResult
from duo_workflow_service.conversation.compaction.integration import (
    maybe_compact_history,
)
from duo_workflow_service.conversation.trimmer import TrimResult


class TestMaybeCompactHistory:
    """Test suite for the ``maybe_compact_history`` shim."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_compactor_none_falls_back_to_trimming(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """When compactor is None, should fall back to token-based trimming."""
        messages = [HumanMessage(content="test")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

        result, compaction_result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        assert result == messages
        assert compaction_result is None
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="test_agent",
            max_context_tokens=400_000,
        )

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_compactor_present_uses_compaction_then_trim(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """When a compactor is provided, compaction runs first, then LegacyTrimOptimizer runs as the terminal safety net
        (no-op when history is within budget)."""
        messages = [HumanMessage(content="test")]
        compacted_messages = [HumanMessage(content="compacted")]

        mock_get_max_context.return_value = 400_000
        # Trim is a no-op because compacted history is within budget
        mock_trim.return_value = TrimResult(
            messages=compacted_messages, was_trimmed=False
        )

        compaction_result_obj = CompactionResult(
            messages=compacted_messages,
            was_modified=True,
        )
        mock_compactor = MagicMock()
        mock_compactor.optimize = AsyncMock(return_value=compaction_result_obj)

        result, returned_result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        assert result == compacted_messages
        assert returned_result is compaction_result_obj
        mock_compactor.optimize.assert_called_once_with(messages)
        # Terminal LegacyTrimOptimizer always runs; here it is a no-op
        mock_trim.assert_called_once_with(
            messages=compacted_messages,
            component_name="test_agent",
            max_context_tokens=400_000,
        )

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_compaction_returns_original_when_not_compacted(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """When compaction decides not to compact, should return original messages and the no-op result."""
        messages = [HumanMessage(content="test")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

        noop_result = CompactionResult(messages=messages, was_modified=False)
        mock_compactor = MagicMock()
        mock_compactor.optimize = AsyncMock(return_value=noop_result)

        result, returned_result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        assert result == messages
        assert returned_result is noop_result
        assert returned_result.was_compacted is False
        mock_compactor.optimize.assert_called_once_with(messages)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_empty_history_with_compactor_none(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """Empty history with no compactor should still call trimming."""
        messages = []
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

        result, compaction_result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        assert result == []
        assert compaction_result is None
        mock_trim.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_legacy_trim_event_fires_when_trimmed(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """Should fire legacy_trim_executed event when trim actually trims.

        The event now fires from inside ``LegacyTrimOptimizer.optimize`` via
        the DI-injected ``InternalEventsClient``.
        """
        messages = [HumanMessage(content="test")]
        trimmed = [HumanMessage(content="trimmed")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed,
            was_trimmed=True,
            tokens_before=300_000,
            tokens_after=200_000,
            messages_before=50,
            messages_after=30,
            token_budget=280_000,
            max_context_tokens=400_000,
            duration_ms=12.5,
        )

        mock_events_client = MagicMock()

        result, compaction_result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
            internal_events_client=mock_events_client,
        )

        assert result == trimmed
        assert compaction_result is None
        mock_events_client.track_event.assert_called_once()
        call_kwargs = mock_events_client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == "duo_workflow_legacy_trim_executed"
        assert call_kwargs["category"] == "legacy_trimmer"
        additional_props = call_kwargs["additional_properties"]
        assert additional_props.label == "test_agent"
        assert additional_props.extra["tokens_before"] == 300_000
        assert additional_props.extra["tokens_after"] == 200_000
        assert additional_props.extra["tokens_removed"] == 100_000
        assert additional_props.extra["messages_before"] == 50
        assert additional_props.extra["messages_after"] == 30
        assert additional_props.extra["token_budget"] == 280_000
        assert additional_props.extra["max_context_tokens"] == 400_000
        assert additional_props.extra["duration_ms"] == 12.5

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_legacy_trim_event_does_not_fire_when_not_trimmed(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """Should NOT fire event when trim short-circuits (below threshold)."""
        messages = [HumanMessage(content="test")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

        mock_events_client = MagicMock()

        result, compaction_result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
            internal_events_client=mock_events_client,
        )

        assert result == messages
        assert compaction_result is None
        mock_events_client.track_event.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_no_trim_event_when_compactor_present_and_trim_is_noop(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """No legacy-trim event fires when compactor is present and trim is a no-op (compacted history is within
        budget)."""
        messages = [HumanMessage(content="test")]
        compacted_messages = [HumanMessage(content="compacted")]

        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=compacted_messages, was_trimmed=False
        )

        mock_compactor = MagicMock()
        mock_compactor.optimize = AsyncMock(
            return_value=CompactionResult(
                messages=compacted_messages,
                was_modified=True,
            )
        )
        mock_events_client = MagicMock()

        result, compaction_result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
            internal_events_client=mock_events_client,
        )

        assert result == compacted_messages
        assert compaction_result is not None
        assert compaction_result.was_compacted is True
        # Trim ran but was a no-op, so no event was fired
        mock_trim.assert_called_once()
        mock_events_client.track_event.assert_not_called()


class TestMaybeCompactHistoryBehavioral:
    """Behavioral tests for the two safety-net scenarios the terminal LegacyTrimOptimizer covers."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_over_budget_below_message_gate_still_trimmed(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """Scenario (a): compaction is a no-op (below message-count gate) but history is over-budget.

        The terminal LegacyTrimOptimizer must trim it.
        """
        messages = [HumanMessage(content="test")]
        trimmed_messages = [HumanMessage(content="trimmed")]

        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed_messages,
            was_trimmed=True,
            tokens_before=300_000,
            tokens_after=200_000,
        )

        # Compaction is a no-op (e.g. should_compact() returned False)
        noop_result = CompactionResult(messages=messages, was_modified=False)
        mock_compactor = MagicMock()
        mock_compactor.optimize = AsyncMock(return_value=noop_result)

        result, returned_result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        # Terminal trim must have caught the over-budget history
        assert result == trimmed_messages
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="test_agent",
            max_context_tokens=400_000,
        )
        # CompactionResult is still returned (no-op result)
        assert returned_result is noop_result

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_compaction_failure_falls_back_to_trim(
        self,
        mock_get_max_context,
        mock_trim,
    ):
        """Scenario (b): compaction summarizer raises an exception; compact() catches it and returns history unchanged
        (was_modified=False, error set).

        The terminal LegacyTrimOptimizer must then trim the over-budget history.
        """
        messages = [HumanMessage(content="test")]
        trimmed_messages = [HumanMessage(content="trimmed after compaction failure")]

        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed_messages,
            was_trimmed=True,
            tokens_before=350_000,
            tokens_after=200_000,
        )

        # Simulate compact() catching an exception and returning history unchanged
        error_result = CompactionResult(
            messages=messages,
            was_modified=False,
            error=RuntimeError("LLM call failed"),
        )
        mock_compactor = MagicMock()
        mock_compactor.optimize = AsyncMock(return_value=error_result)

        result, returned_result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        # Terminal trim must have caught the over-budget history
        assert result == trimmed_messages
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="test_agent",
            max_context_tokens=400_000,
        )
        # The CompactionResult (with error) is still returned
        assert returned_result is error_result


class TestMaybeCompactHistorySelfHosted:
    """Self-hosted mode no longer affects the dispatch decision.

    Selection is purely structural (compactor is None vs. provided). These
    tests preserve coverage of the contract without depending on the legacy
    ``get_config()`` call that the shim no longer reads.
    """

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_compactor_provided_runs_compaction(
        self, mock_get_max_context, mock_trim
    ):
        """When a compactor is provided, compaction runs unconditionally."""
        messages = [HumanMessage(content="test")]
        compacted_messages = [HumanMessage(content="compacted")]

        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=compacted_messages, was_trimmed=False
        )

        mock_compactor = MagicMock()
        mock_compactor.optimize = AsyncMock(
            return_value=CompactionResult(
                messages=compacted_messages,
                was_modified=True,
            )
        )

        result, compaction_result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
            internal_events_client=MagicMock(),
        )

        assert result == compacted_messages
        assert compaction_result is not None
        mock_compactor.optimize.assert_called_once_with(messages)
