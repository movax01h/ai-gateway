"""Tests for integration.py maybe_compact_history function."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.conversation.compaction import CompactionResult
from duo_workflow_service.conversation.compaction.integration import (
    maybe_compact_history,
)


class TestMaybeCompactHistory:
    """Test suite for maybe_compact_history function."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.integration.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.get_model_max_context_token_limit"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    async def test_compactor_none_falls_back_to_trimming(
        self, mock_is_feature_enabled, mock_get_max_context, mock_trim
    ):
        """When compactor is None, should fall back to token-based trimming."""
        messages = [HumanMessage(content="test")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = messages

        result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
        )

        assert result == messages
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="test_agent",
            max_context_tokens=400_000,
        )
        mock_is_feature_enabled.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.integration.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.get_model_max_context_token_limit"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    async def test_feature_flag_disabled_falls_back_to_trimming(
        self, mock_is_feature_enabled, mock_get_max_context, mock_trim
    ):
        """When feature flag is disabled, should fall back to token-based trimming."""
        messages = [HumanMessage(content="test")]
        mock_is_feature_enabled.return_value = False
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = messages

        mock_compactor = AsyncMock()

        result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
        )

        assert result == messages
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="test_agent",
            max_context_tokens=400_000,
        )
        mock_compactor.compact.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.integration.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    async def test_compactor_enabled_and_feature_flag_enabled_uses_compaction(
        self, mock_is_feature_enabled, mock_trim
    ):
        """When compactor exists and feature flag enabled, should use compaction."""
        messages = [HumanMessage(content="test")]
        compacted_messages = [HumanMessage(content="compacted")]
        mock_is_feature_enabled.return_value = True

        mock_compactor = AsyncMock()
        mock_compactor.compact.return_value = CompactionResult(
            messages=compacted_messages,
            was_compacted=True,
        )

        result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
        )

        assert result == compacted_messages
        mock_compactor.compact.assert_called_once_with(messages=messages)
        mock_trim.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    async def test_compaction_returns_original_when_not_compacted(
        self, mock_is_feature_enabled
    ):
        """When compaction decides not to compact, should return original messages."""
        messages = [HumanMessage(content="test")]
        mock_is_feature_enabled.return_value = True

        mock_compactor = AsyncMock()
        mock_compactor.compact.return_value = CompactionResult(
            messages=messages,
            was_compacted=False,
        )

        result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
        )

        assert result == messages
        mock_compactor.compact.assert_called_once_with(messages=messages)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.integration.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.get_model_max_context_token_limit"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    async def test_empty_history_with_compactor_none(
        self, mock_is_feature_enabled, mock_get_max_context, mock_trim
    ):
        """Empty history with no compactor should still call trimming."""
        messages = []
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = messages

        result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
        )

        assert result == []
        mock_trim.assert_called_once()
