"""Tests for ``validate_history_optimizer_configs``."""

import pytest

from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    LegacyTrimConfig,
    ToolResultPrunerConfig,
)
from duo_workflow_service.conversation.history_optimizer.validation import (
    validate_history_optimizer_configs,
)


class TestValidateHistoryOptimizerConfigs:
    def test_empty_list_ok(self):
        validate_history_optimizer_configs([])

    def test_single_legacy_trim_ok(self):
        validate_history_optimizer_configs([LegacyTrimConfig()])

    def test_single_compaction_ok(self):
        validate_history_optimizer_configs([CompactionConfig()])

    def test_single_tool_result_pruner_ok(self):
        validate_history_optimizer_configs([ToolResultPrunerConfig()])

    def test_compaction_and_legacy_trim_rejected(self):
        with pytest.raises(ValueError, match="at most one of"):
            validate_history_optimizer_configs([CompactionConfig(), LegacyTrimConfig()])

    def test_duplicate_compaction_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_history_optimizer_configs([CompactionConfig(), CompactionConfig()])

    def test_duplicate_legacy_trim_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_history_optimizer_configs([LegacyTrimConfig(), LegacyTrimConfig()])

    def test_pruner_then_compaction_ok(self):
        validate_history_optimizer_configs(
            [ToolResultPrunerConfig(), CompactionConfig()]
        )

    def test_compaction_then_pruner_ok(self):
        validate_history_optimizer_configs(
            [CompactionConfig(), ToolResultPrunerConfig()]
        )

    def test_pruner_then_legacy_trim_ok(self):
        validate_history_optimizer_configs(
            [ToolResultPrunerConfig(), LegacyTrimConfig()]
        )

    def test_duplicate_pruner_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            validate_history_optimizer_configs(
                [ToolResultPrunerConfig(), ToolResultPrunerConfig()]
            )
