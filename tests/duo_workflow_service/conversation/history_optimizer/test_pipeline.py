"""Tests for ``HistoryOptimizerPipeline``."""

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from duo_workflow_service.conversation.history_optimizer.base import HistoryOptimizer
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    OptimizationResult,
)


class _FakeOptimizer(HistoryOptimizer):
    def __init__(
        self,
        *,
        modified: bool,
        replacement: list[BaseMessage] | None = None,
        name: str = "Fake",
    ):
        self._modified = modified
        self._replacement = replacement
        self._name = name
        self.calls: list[list[BaseMessage]] = []

    async def optimize(self, history: list[BaseMessage]) -> OptimizationResult:
        self.calls.append(history)
        messages = self._replacement if self._modified else history
        return OptimizationResult(
            messages=messages if messages is not None else history,
            was_modified=self._modified,
            optimizer_name=self._name,
        )


class TestHistoryOptimizerPipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline_returns_history_and_empty_results(self):
        pipeline = HistoryOptimizerPipeline([])
        history = [HumanMessage(content="x")]
        messages, results = await pipeline.optimize(history)
        assert messages is history
        assert results == []

    @pytest.mark.asyncio
    async def test_single_optimizer_modifies(self):
        replacement = [HumanMessage(content="trimmed")]
        opt = _FakeOptimizer(modified=True, replacement=replacement)
        pipeline = HistoryOptimizerPipeline([opt])

        original = [HumanMessage(content="original")]
        messages, results = await pipeline.optimize(original)

        assert messages == replacement
        assert len(results) == 1
        assert results[0].was_modified is True

    @pytest.mark.asyncio
    async def test_single_optimizer_no_op_preserves_history(self):
        opt = _FakeOptimizer(modified=False)
        pipeline = HistoryOptimizerPipeline([opt])

        original = [HumanMessage(content="original")]
        messages, results = await pipeline.optimize(original)

        assert messages is original
        assert len(results) == 1
        assert results[0].was_modified is False

    @pytest.mark.asyncio
    async def test_two_optimizers_both_modify_chained(self):
        first_out = [HumanMessage(content="first")]
        second_out = [HumanMessage(content="second")]
        first = _FakeOptimizer(modified=True, replacement=first_out, name="First")
        second = _FakeOptimizer(modified=True, replacement=second_out, name="Second")
        pipeline = HistoryOptimizerPipeline([first, second])

        original = [HumanMessage(content="original")]
        messages, results = await pipeline.optimize(original)

        assert messages == second_out
        assert first.calls == [original]
        assert second.calls == [first_out]
        assert [r.optimizer_name for r in results] == ["First", "Second"]

    @pytest.mark.asyncio
    async def test_first_no_op_second_modifies(self):
        replacement = [HumanMessage(content="r")]
        first = _FakeOptimizer(modified=False, name="First")
        second = _FakeOptimizer(modified=True, replacement=replacement, name="Second")
        pipeline = HistoryOptimizerPipeline([first, second])

        original = [HumanMessage(content="original")]
        messages, results = await pipeline.optimize(original)

        assert messages == replacement
        assert first.calls == [original]
        # second received the original (first was no-op)
        assert second.calls == [original]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_results_order_matches_optimizers_order(self):
        opts = [_FakeOptimizer(modified=False, name=f"O{i}") for i in range(3)]
        pipeline = HistoryOptimizerPipeline(opts)
        _, results = await pipeline.optimize([HumanMessage(content="x")])
        assert [r.optimizer_name for r in results] == ["O0", "O1", "O2"]

    def test_optimizers_property(self):
        opts = [_FakeOptimizer(modified=False)]
        pipeline = HistoryOptimizerPipeline(opts)
        assert pipeline.optimizers is opts
