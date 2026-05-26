"""Tests for integration.py maybe_compact_history function."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.conversation.compaction import CompactionResult
from duo_workflow_service.conversation.compaction.integration import (
    maybe_compact_history,
)
from duo_workflow_service.conversation.trimmer import TrimResult


def _mock_config(custom_models_enabled=False):
    """Create a mock Config with custom_models.enabled set."""
    mock = MagicMock()
    mock.custom_models.enabled = custom_models_enabled
    return mock


@patch("duo_workflow_service.conversation.compaction.integration.is_gitlab_team_member")
@patch(
    "duo_workflow_service.conversation.compaction.integration.get_config",
    return_value=_mock_config(custom_models_enabled=False),
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
        self,
        mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """When compactor is None, should fall back to token-based trimming."""
        messages = [HumanMessage(content="test")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

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
        self,
        mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """When feature flag is disabled, should fall back to token-based trimming."""
        messages = [HumanMessage(content="test")]
        mock_is_feature_enabled.return_value = False
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

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
        self,
        mock_is_feature_enabled,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
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
        self,
        mock_is_feature_enabled,
        _mock_get_config,
        _mock_is_gitlab_team_member,
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
        self,
        _mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """Empty history with no compactor should still call trimming."""
        messages = []
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

        result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
        )

        assert result == []
        mock_trim.assert_called_once()

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
    async def test_legacy_trim_event_fires_when_trimmed(
        self,
        mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """Should fire legacy_trim_executed event when trim actually trims.

        The event fires via compactor._internal_events_client when the compactor exists but the feature flag is off
        (fallback to trim).
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

        # The event fires via compactor._internal_events_client, so we
        # need a compactor with the right private attributes set.
        mock_events_client = MagicMock()
        mock_compactor = MagicMock()
        mock_compactor._internal_events_client = mock_events_client
        mock_compactor._workflow_id = "wf-123"
        mock_compactor._workflow_type = "developer"

        # Feature flag is off → falls through to legacy trim
        mock_is_feature_enabled.return_value = False

        result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
        )

        assert result == trimmed
        mock_events_client.track_event.assert_called_once()
        call_kwargs = mock_events_client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == "duo_workflow_legacy_trim_executed"
        assert call_kwargs["category"] == "legacy_trimmer"
        additional_props = call_kwargs["additional_properties"]
        assert additional_props.label == "test_agent"
        assert additional_props.property == "workflow_id"
        assert additional_props.value == "wf-123"
        assert additional_props.extra["tokens_before"] == 300_000
        assert additional_props.extra["tokens_after"] == 200_000
        assert additional_props.extra["tokens_removed"] == 100_000

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
    async def test_legacy_trim_event_does_not_fire_when_not_trimmed(
        self,
        mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """Should NOT fire event when trim short-circuits (below threshold)."""
        messages = [HumanMessage(content="test")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(messages=messages, was_trimmed=False)

        mock_events_client = MagicMock()
        mock_compactor = MagicMock()
        mock_compactor._internal_events_client = mock_events_client
        mock_compactor._workflow_id = "wf-123"
        mock_compactor._workflow_type = "developer"

        # Feature flag is off → falls through to legacy trim
        mock_is_feature_enabled.return_value = False

        result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
        )

        assert result == messages
        mock_events_client.track_event.assert_not_called()

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
    async def test_legacy_trim_event_does_not_fire_without_client(
        self,
        mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """Should NOT fire event when compactor._internal_events_client is None."""
        messages = [HumanMessage(content="test")]
        trimmed = [HumanMessage(content="trimmed")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed,
            was_trimmed=True,
            tokens_before=300_000,
            tokens_after=200_000,
        )

        # Compactor exists but has no internal events client
        mock_compactor = MagicMock()
        mock_compactor._internal_events_client = None

        # Feature flag is off → falls through to legacy trim
        mock_is_feature_enabled.return_value = False

        result = await maybe_compact_history(
            compactor=mock_compactor,
            history=messages,
            agent_name="test_agent",
        )

        assert result == trimmed

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
    async def test_no_event_when_compactor_is_none(
        self,
        _mock_is_feature_enabled,
        mock_get_max_context,
        mock_trim,
        _mock_get_config,
        _mock_is_gitlab_team_member,
    ):
        """No event fires when compactor is None (no client to fire with)."""
        messages = [HumanMessage(content="test")]
        trimmed = [HumanMessage(content="trimmed")]
        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed,
            was_trimmed=True,
            tokens_before=300_000,
            tokens_after=200_000,
        )

        result = await maybe_compact_history(
            compactor=None,
            history=messages,
            agent_name="test_agent",
        )

        # compactor is None → event block is skipped entirely
        assert result == trimmed


@patch("duo_workflow_service.conversation.compaction.integration.is_gitlab_team_member")
class TestMaybeCompactHistorySelfHosted:
    """Tests for compaction behavior in self-hosted mode (AIGW_CUSTOM_MODELS__ENABLED=true)."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.conversation.compaction.integration.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.get_config",
        return_value=_mock_config(custom_models_enabled=True),
    )
    async def test_self_hosted_mode_allows_compaction(
        self,
        _mock_get_config,
        mock_is_feature_enabled,
        mock_trim,
        _mock_is_gitlab_team_member,
    ):
        """When custom_models.enabled=True, compaction runs normally with FF on and compactor provided."""
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
        "duo_workflow_service.conversation.compaction.integration.apply_token_based_trim"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.is_feature_enabled"
    )
    @patch(
        "duo_workflow_service.conversation.compaction.integration.get_config",
        return_value=_mock_config(custom_models_enabled=False),
    )
    async def test_non_self_hosted_mode_allows_compaction(
        self,
        _mock_get_config,
        mock_is_feature_enabled,
        mock_trim,
        _mock_is_gitlab_team_member,
    ):
        """When custom_models.enabled=False, compaction works normally."""
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
