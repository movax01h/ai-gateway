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
from duo_workflow_service.conversation.history_optimizer.schema import CompactionConfig


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
    def test_default_uses_legacy_trim(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        assert isinstance(pipeline.optimizers[0], LegacyTrimOptimizer)

    def test_compaction_false_yields_legacy_trim(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            compaction=False,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        assert isinstance(pipeline.optimizers[0], LegacyTrimOptimizer)

    def test_compaction_true_yields_default_compaction_optimizer(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            compaction=True,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        optimizer = pipeline.optimizers[0]
        assert isinstance(optimizer, CompactionOptimizer)
        assert optimizer._config == CompactionConfig()

    def test_compaction_config_yields_compaction_optimizer_with_config(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        cfg = CompactionConfig(trim_threshold=0.5)
        pipeline = build_history_optimizer_pipeline(
            compaction=cfg,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        optimizer = pipeline.optimizers[0]
        assert isinstance(optimizer, CompactionOptimizer)
        assert optimizer._config is cfg

    def test_compaction_optimizer_receives_flow_context_fields(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            compaction=CompactionConfig(),
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

    def test_legacy_trim_receives_agent_name_and_events_client(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            compaction=False,
            flow_context=flow_context,
            agent_name="my_agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        optimizer = pipeline.optimizers[0]
        assert isinstance(optimizer, LegacyTrimOptimizer)
        assert optimizer._agent_name == "my_agent"
        assert optimizer._internal_events_client is mock_internal_events_client
