"""Tests for ``build_history_optimizer_pipeline`` and ``FlowContext``."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

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
    CompactionResult,
)
from duo_workflow_service.conversation.trimmer import TrimResult


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
    def test_default_uses_legacy_trim_only(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        """When compaction is not specified, the pipeline contains only LegacyTrimOptimizer."""
        pipeline = build_history_optimizer_pipeline(
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        assert isinstance(pipeline.optimizers[0], LegacyTrimOptimizer)

    def test_compaction_false_yields_legacy_trim_only(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        """Explicitly disabled compaction yields a single LegacyTrimOptimizer."""
        pipeline = build_history_optimizer_pipeline(
            compaction=False,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 1
        assert isinstance(pipeline.optimizers[0], LegacyTrimOptimizer)

    @pytest.mark.parametrize(
        "compaction_arg",
        [True, CompactionConfig()],
        ids=["bool_true", "default_config"],
    )
    def test_compaction_enabled_yields_compaction_then_legacy_trim(
        self,
        compaction_arg,
        flow_context,
        mock_prompt_registry,
        mock_internal_events_client,
    ):
        """Enabled compaction yields [CompactionOptimizer, LegacyTrimOptimizer]."""
        pipeline = build_history_optimizer_pipeline(
            compaction=compaction_arg,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        assert len(pipeline.optimizers) == 2
        assert isinstance(pipeline.optimizers[0], CompactionOptimizer)
        assert isinstance(pipeline.optimizers[1], LegacyTrimOptimizer)

    def test_compaction_true_uses_default_config(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        pipeline = build_history_optimizer_pipeline(
            compaction=True,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
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

    def test_terminal_legacy_trim_receives_agent_name_and_events_client(
        self, flow_context, mock_prompt_registry, mock_internal_events_client
    ):
        """The terminal LegacyTrimOptimizer (when compaction is enabled) gets the right args."""
        pipeline = build_history_optimizer_pipeline(
            compaction=True,
            flow_context=flow_context,
            agent_name="my_agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        trim_optimizer = pipeline.optimizers[1]
        assert isinstance(trim_optimizer, LegacyTrimOptimizer)
        assert trim_optimizer._agent_name == "my_agent"
        assert trim_optimizer._internal_events_client is mock_internal_events_client


class TestBuildHistoryOptimizerPipelineBehavioral:
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
        flow_context,
        mock_prompt_registry,
        mock_internal_events_client,
    ):
        """Scenario (a): history is over-budget but has fewer messages than max_recent_messages, so should_compact()
        returns False and compaction is a no-op.

        The terminal LegacyTrimOptimizer must still trim it.
        """
        # Fewer messages than the default max_recent_messages (10)
        messages = [HumanMessage(content=f"msg {i}") for i in range(5)]
        trimmed_messages = [HumanMessage(content="trimmed")]

        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed_messages,
            was_trimmed=True,
            tokens_before=300_000,
            tokens_after=200_000,
        )

        # CompactionOptimizer.optimize() short-circuits via should_compact() → False
        mock_compaction_optimizer = MagicMock(spec=CompactionOptimizer)
        mock_compaction_optimizer.optimize = AsyncMock(
            return_value=CompactionResult(messages=messages, was_modified=False)
        )

        pipeline = build_history_optimizer_pipeline(
            compaction=True,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        # Replace the real CompactionOptimizer with our mock
        pipeline._optimizers[0] = mock_compaction_optimizer

        final_messages, _results = await pipeline.optimize(messages)

        assert final_messages == trimmed_messages
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="agent",
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
    async def test_compaction_failure_falls_back_to_trim(
        self,
        mock_get_max_context,
        mock_trim,
        flow_context,
        mock_prompt_registry,
        mock_internal_events_client,
    ):
        """Scenario (b): compaction summarizer raises an exception; compact() catches it and returns history unchanged
        (was_modified=False).

        The terminal LegacyTrimOptimizer must then trim the over-budget history.
        """
        messages = [HumanMessage(content=f"msg {i}") for i in range(20)]
        trimmed_messages = [HumanMessage(content="trimmed after compaction failure")]

        mock_get_max_context.return_value = 400_000
        mock_trim.return_value = TrimResult(
            messages=trimmed_messages,
            was_trimmed=True,
            tokens_before=350_000,
            tokens_after=200_000,
        )

        # Simulate compact() catching an exception and returning history unchanged
        mock_compaction_optimizer = MagicMock(spec=CompactionOptimizer)
        mock_compaction_optimizer.optimize = AsyncMock(
            return_value=CompactionResult(
                messages=messages,
                was_modified=False,
                error=RuntimeError("LLM call failed"),
            )
        )

        pipeline = build_history_optimizer_pipeline(
            compaction=True,
            flow_context=flow_context,
            agent_name="agent",
            prompt_registry=mock_prompt_registry,
            internal_events_client=mock_internal_events_client,
        )
        # Replace the real CompactionOptimizer with our mock
        pipeline._optimizers[0] = mock_compaction_optimizer

        final_messages, _results = await pipeline.optimize(messages)

        assert final_messages == trimmed_messages
        mock_trim.assert_called_once_with(
            messages=messages,
            component_name="agent",
            max_context_tokens=400_000,
        )
