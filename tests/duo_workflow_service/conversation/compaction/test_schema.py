"""Tests for compaction schema classes."""

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from duo_workflow_service.conversation.compaction import (
    CompactionConfig,
    CompactionResult,
    MessageSlices,
)


class TestCompactionConfig:
    """Test suite for CompactionConfig dataclass."""

    def test_default_values(self):
        """Should have expected default values."""
        config = CompactionConfig()
        assert config.max_recent_messages == 10
        assert config.recent_messages_token_budget == 40_000
        assert config.trim_threshold == 0.7

    def test_frozen(self):
        """Should be immutable (frozen model)."""
        config = CompactionConfig()
        with pytest.raises(ValidationError):
            config.max_recent_messages = 20

    def test_custom_values(self):
        """Should accept custom values."""
        config = CompactionConfig(
            max_recent_messages=5,
            recent_messages_token_budget=20_000,
            trim_threshold=0.5,
        )
        assert config.max_recent_messages == 5
        assert config.recent_messages_token_budget == 20_000
        assert config.trim_threshold == 0.5


class TestCompactionResult:
    """Test suite for CompactionResult dataclass."""

    def test_default_values(self):
        """Should have expected default values for optional fields."""
        result = CompactionResult(messages=[], was_compacted=False)
        assert result.tokens_before == 0
        assert result.tokens_after == 0
        assert result.messages_summarized == 0
        assert result.compaction_input_tokens == 0
        assert result.compaction_output_tokens == 0
        assert result.error is None

    def test_with_messages(self):
        """Should store messages correctly."""
        messages = [HumanMessage(content="test")]
        result = CompactionResult(messages=messages, was_compacted=True)
        assert result.messages == messages

    def test_with_error(self):
        """Should store error correctly."""
        error = Exception("test error")
        result = CompactionResult(messages=[], was_compacted=False, error=error)
        assert result.error is error

    def test_with_token_counts(self):
        """Should store token counts correctly."""
        result = CompactionResult(
            messages=[],
            was_compacted=True,
            tokens_before=1000,
            tokens_after=500,
            messages_summarized=10,
            compaction_input_tokens=800,
            compaction_output_tokens=150,
        )
        assert result.tokens_before == 1000
        assert result.tokens_after == 500
        assert result.messages_summarized == 10
        assert result.compaction_input_tokens == 800
        assert result.compaction_output_tokens == 150


class TestMessageSlices:
    """Test suite for MessageSlices dataclass."""

    def test_dataclass_fields(self):
        """Should have expected fields."""
        slices = MessageSlices(
            leading_context=[],
            to_summarize=[],
            recent_to_keep=[],
        )
        assert hasattr(slices, "leading_context")
        assert hasattr(slices, "to_summarize")
        assert hasattr(slices, "recent_to_keep")

    def test_with_messages(self):
        """Should store messages in correct fields."""
        leading = [HumanMessage(content="initial")]
        to_summarize = [HumanMessage(content="middle")]
        recent = [HumanMessage(content="recent")]

        slices = MessageSlices(
            leading_context=leading,
            to_summarize=to_summarize,
            recent_to_keep=recent,
        )

        assert slices.leading_context == leading
        assert slices.to_summarize == to_summarize
        assert slices.recent_to_keep == recent
