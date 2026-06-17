"""Tests for ``build_history_optimizer_pipeline`` and ``FlowContext``."""

from unittest.mock import MagicMock

import pytest

from duo_workflow_service.conversation.history_optimizer.builder import (
    FlowContext,
    build_history_optimizer_pipeline,
)
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    CompactionOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.optimizers.legacy_trim import (
    LegacyTrimOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    LegacyTrimConfig,
    ToolResultPrunerConfig,
)


@pytest.fixture(name="flow_context")
def flow_context_fixture():
    return FlowContext(
        flow_id="flow-1",
        flow_type="chat",
        user=MagicMock(),
    )


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture():
    return MagicMock()


@pytest.fixture(name="mock_internal_events_client")
def mock_internal_events_client_fixture():
    return MagicMock()


class TestFlowContext:
    def test_carries_fields(self):
        user = MagicMock()
        ctx = FlowContext(flow_id="f1", flow_type="t1", user=user)
        assert ctx.flow_id == "f1"
        assert ctx.flow_type == "t1"
        assert ctx.user is user

    def test_is_frozen(self):
        ctx = FlowContext(flow_id="f", flow_type="t", user=MagicMock())
        with pytest.raises(Exception):
            ctx.flow_id = "f2"


class TestBuildHistoryOptimizerPipeline:
    def test_empty_configs_yield_empty_pipeline(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            [],
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert pipeline.optimizers == []

    def test_compaction_config_builds_compaction_optimizer(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        cfg = CompactionConfig()
        pipeline = build_history_optimizer_pipeline(
            [cfg],
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        assert isinstance(pipeline.optimizers[0], CompactionOptimizer)

    def test_legacy_trim_config_builds_legacy_trim_optimizer(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        cfg = LegacyTrimConfig()
        pipeline = build_history_optimizer_pipeline(
            [cfg],
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        assert isinstance(pipeline.optimizers[0], LegacyTrimOptimizer)

    def test_tool_result_pruner_config_raises_not_implemented(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        with pytest.raises(NotImplementedError, match="ToolResultPruner"):
            build_history_optimizer_pipeline(
                [ToolResultPrunerConfig()],
                flow_context=flow_context,
                agent_name="agent",
                prompt_registry=mock_prompt_registry,
                internal_events_client=mock_internal_events_client,
            )

    def test_validation_runs_before_construction(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        with pytest.raises(ValueError, match="at most one of"):
            build_history_optimizer_pipeline(
                [CompactionConfig(), LegacyTrimConfig()],
                flow_context=flow_context,
                agent_name="agent",
                prompt_registry=mock_prompt_registry,
                internal_events_client=mock_internal_events_client,
            )

    def test_unknown_config_type_raises_value_error(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        class UnknownConfig:
            pass

        with pytest.raises(ValueError, match="Unknown HistoryOptimizerConfig"):
            build_history_optimizer_pipeline(
                [UnknownConfig()],
                flow_context=flow_context,
                agent_name="agent",
                prompt_registry=mock_prompt_registry,
                internal_events_client=mock_internal_events_client,
            )

    def test_compaction_optimizer_receives_flow_context_fields(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            [CompactionConfig()],
            flow_context=flow_context,
            agent_name="my_agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        optimizer = pipeline.optimizers[0]
        assert optimizer._workflow_id == flow_context.flow_id
        assert optimizer._workflow_type == flow_context.flow_type
        assert optimizer._user is flow_context.user
        assert optimizer._agent_name == "my_agent"
        assert optimizer._prompt_registry is mock_prompt_registry
        assert optimizer._internal_events_client is mock_internal_events_client
