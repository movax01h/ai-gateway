"""Tests for ``LegacyTrimOptimizer``."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.conversation.history_optimizer.optimizers.legacy_trim import (
    LegacyTrimOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.schema import TrimResult
from duo_workflow_service.conversation.trimmer import TrimResult as InnerTrimResult


@pytest.fixture(name="mock_internal_events_client")
def mock_internal_events_client_fixture():
    return MagicMock()


@pytest.fixture(name="optimizer")
def optimizer_fixture(mock_internal_events_client):
    return LegacyTrimOptimizer(
        agent_name="test_agent",
        internal_events_client=mock_internal_events_client,
    )


class TestLegacyTrimOptimizer:
    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_under_budget_returns_was_modified_false(
        self,
        mock_get_max_context,
        mock_apply_trim,
        optimizer,
        mock_internal_events_client,
    ):
        messages = [HumanMessage(content="x")]
        mock_get_max_context.return_value = 400_000
        mock_apply_trim.return_value = InnerTrimResult(
            messages=messages,
            was_trimmed=False,
            tokens_before=100,
            token_budget=280_000,
            max_context_tokens=400_000,
        )

        result = await optimizer.optimize(messages)

        assert isinstance(result, TrimResult)
        assert result.was_modified is False
        assert result.messages == messages
        assert result.token_budget == 280_000
        assert result.max_context_tokens == 400_000
        mock_internal_events_client.track_event.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_over_budget_returns_was_modified_true(
        self,
        mock_get_max_context,
        mock_apply_trim,
        optimizer,
    ):
        messages = [HumanMessage(content="x")]
        trimmed = [HumanMessage(content="trimmed")]
        mock_get_max_context.return_value = 400_000
        mock_apply_trim.return_value = InnerTrimResult(
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

        result = await optimizer.optimize(messages)

        assert result.was_modified is True
        assert result.messages == trimmed
        assert result.tokens_before == 300_000
        assert result.tokens_after == 200_000
        assert result.duration_ms == 12.5

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_fires_legacy_trim_event_when_trimmed(
        self,
        mock_get_max_context,
        mock_apply_trim,
        optimizer,
        mock_internal_events_client,
    ):
        messages = [HumanMessage(content="x")]
        mock_get_max_context.return_value = 400_000
        mock_apply_trim.return_value = InnerTrimResult(
            messages=[HumanMessage(content="t")],
            was_trimmed=True,
            tokens_before=300_000,
            tokens_after=200_000,
            messages_before=50,
            messages_after=30,
            token_budget=280_000,
            max_context_tokens=400_000,
            duration_ms=12.5,
        )

        await optimizer.optimize(messages)

        mock_internal_events_client.track_event.assert_called_once()
        kwargs = mock_internal_events_client.track_event.call_args.kwargs
        assert kwargs["event_name"] == "duo_workflow_legacy_trim_executed"
        assert kwargs["category"] == "legacy_trimmer"
        props = kwargs["additional_properties"]
        assert props.label == "test_agent"
        assert props.extra["tokens_before"] == 300_000
        assert props.extra["tokens_after"] == 200_000
        assert props.extra["tokens_removed"] == 100_000
        assert props.extra["messages_before"] == 50
        assert props.extra["messages_after"] == 30
        assert props.extra["token_budget"] == 280_000
        assert props.extra["max_context_tokens"] == 400_000
        assert props.extra["duration_ms"] == 12.5

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.history_optimizer.optimizers."
        "legacy_trim.get_current_model_max_context_token_limit"
    )
    async def test_ui_chat_logs_always_empty(
        self,
        mock_get_max_context,
        mock_apply_trim,
        optimizer,
    ):
        mock_get_max_context.return_value = 400_000
        mock_apply_trim.return_value = InnerTrimResult(
            messages=[],
            was_trimmed=True,
            tokens_before=100,
            tokens_after=50,
        )
        result = await optimizer.optimize([])
        assert result.ui_chat_logs == []
