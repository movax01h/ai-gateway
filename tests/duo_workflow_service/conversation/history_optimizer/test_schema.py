"""Tests for ``history_optimizer.schema`` types."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, TypeAdapter, ValidationError

from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    CompactionResult,
    HistoryOptimizerConfig,
    LegacyTrimConfig,
    OptimizationResult,
    ToolResultPrunerConfig,
    TrimResult,
)


class TestOptimizationResult:
    def test_default_values(self):
        result = OptimizationResult()
        assert not result.messages
        assert result.was_modified is False
        assert result.tokens_before == 0
        assert result.tokens_after == 0
        assert result.duration_ms == 0.0
        assert result.optimizer_name == ""
        assert not result.ui_chat_logs

    def test_custom_values(self):
        messages = [HumanMessage(content="x")]
        result = OptimizationResult(
            messages=messages,
            was_modified=True,
            tokens_before=100,
            tokens_after=50,
            duration_ms=12.5,
            optimizer_name="X",
        )
        assert result.messages is messages
        assert result.was_modified is True
        assert result.tokens_before == 100
        assert result.tokens_after == 50
        assert result.duration_ms == 12.5
        assert result.optimizer_name == "X"


class TestCompactionResult:
    """``CompactionResult`` is constructed with ``was_modified``; ``was_compacted`` is a read-only legacy alias."""

    def test_subclasses_optimization_result(self):
        result = CompactionResult()
        assert isinstance(result, OptimizationResult)

    def test_default_values(self):
        result = CompactionResult()
        assert result.was_modified is False
        assert result.was_compacted is False
        assert result.summary is None
        assert result.messages_summarized == 0
        assert result.compaction_input_tokens == 0
        assert result.compaction_output_tokens == 0
        assert result.error is None

    def test_was_modified_true_exposes_was_compacted_true(self):
        result = CompactionResult(was_modified=True)
        assert result.was_modified is True
        assert result.was_compacted is True

    def test_was_modified_false_exposes_was_compacted_false(self):
        result = CompactionResult(messages=[], was_modified=False)
        assert result.was_modified is False
        assert result.was_compacted is False

    def test_was_compacted_is_read_only(self):
        result = CompactionResult(was_modified=True)
        # ``was_compacted`` is a property, not a settable attribute.
        with pytest.raises(AttributeError):
            result.was_compacted = False

    def test_carries_summary_and_token_fields(self):
        summary = AIMessage(content="s")
        err = RuntimeError("e")
        result = CompactionResult(
            messages=[],
            was_modified=True,
            tokens_before=100,
            tokens_after=50,
            summary=summary,
            messages_summarized=3,
            compaction_input_tokens=80,
            compaction_output_tokens=20,
            error=err,
        )
        assert result.summary is summary
        assert result.messages_summarized == 3
        assert result.compaction_input_tokens == 80
        assert result.compaction_output_tokens == 20
        assert result.error is err


class TestTrimResult:
    def test_subclasses_optimization_result(self):
        result = TrimResult()
        assert isinstance(result, OptimizationResult)

    def test_default_values(self):
        result = TrimResult()
        assert result.token_budget == 0
        assert result.max_context_tokens == 0
        assert result.was_modified is False

    def test_with_fields(self):
        result = TrimResult(
            messages=[HumanMessage(content="x")],
            was_modified=True,
            tokens_before=100,
            tokens_after=50,
            duration_ms=12.5,
            optimizer_name="LegacyTrimOptimizer",
            token_budget=70,
            max_context_tokens=100,
        )
        assert result.token_budget == 70
        assert result.max_context_tokens == 100


class TestConfigs:
    def test_compaction_config_defaults(self):
        cfg = CompactionConfig()
        assert cfg.type == "compaction"
        assert cfg.max_recent_messages == 10
        assert cfg.trim_threshold == 0.7

    def test_legacy_trim_config_defaults(self):
        cfg = LegacyTrimConfig()
        assert cfg.type == "legacy_trim"

    def test_tool_result_pruner_config_defaults(self):
        cfg = ToolResultPrunerConfig()
        assert cfg.type == "tool_result_pruner"

    def test_compaction_config_is_frozen(self):
        cfg = CompactionConfig()
        with pytest.raises(ValidationError):
            cfg.max_recent_messages = 99

    def test_compaction_config_rejects_non_positive_max_recent_messages(self):
        with pytest.raises(
            ValidationError, match="max_recent_messages must be positive"
        ):
            CompactionConfig(max_recent_messages=0)
        with pytest.raises(
            ValidationError, match="max_recent_messages must be positive"
        ):
            CompactionConfig(max_recent_messages=-1)

    def test_compaction_config_rejects_non_positive_token_budget(self):
        with pytest.raises(
            ValidationError, match="recent_messages_token_budget must be positive"
        ):
            CompactionConfig(recent_messages_token_budget=0)
        with pytest.raises(
            ValidationError, match="recent_messages_token_budget must be positive"
        ):
            CompactionConfig(recent_messages_token_budget=-100)

    def test_compaction_config_rejects_invalid_trim_threshold(self):
        # Must be in (0, 1].
        with pytest.raises(ValidationError, match="trim_threshold must be between"):
            CompactionConfig(trim_threshold=0)
        with pytest.raises(ValidationError, match="trim_threshold must be between"):
            CompactionConfig(trim_threshold=-0.1)
        with pytest.raises(ValidationError, match="trim_threshold must be between"):
            CompactionConfig(trim_threshold=1.1)

    def test_legacy_trim_config_is_frozen(self):
        cfg = LegacyTrimConfig()
        with pytest.raises(ValidationError):
            cfg.type = "compaction"


class _ConfigWrapper(BaseModel):
    """Wrapper model used to exercise the discriminated union parsing."""

    config: HistoryOptimizerConfig


class TestDiscriminatedUnionParsing:
    def test_parses_compaction(self):
        adapter = TypeAdapter(HistoryOptimizerConfig)
        parsed = adapter.validate_python({"type": "compaction", "trim_threshold": 0.5})
        assert isinstance(parsed, CompactionConfig)
        assert parsed.trim_threshold == 0.5

    def test_parses_legacy_trim(self):
        adapter = TypeAdapter(HistoryOptimizerConfig)
        parsed = adapter.validate_python({"type": "legacy_trim"})
        assert isinstance(parsed, LegacyTrimConfig)

    def test_parses_tool_result_pruner(self):
        adapter = TypeAdapter(HistoryOptimizerConfig)
        parsed = adapter.validate_python({"type": "tool_result_pruner"})
        assert isinstance(parsed, ToolResultPrunerConfig)

    def test_unknown_type_rejected(self):
        adapter = TypeAdapter(HistoryOptimizerConfig)
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "not_a_real_optimizer"})

    def test_inside_pydantic_model(self):
        m = _ConfigWrapper.model_validate({"config": {"type": "compaction"}})
        assert isinstance(m.config, CompactionConfig)
