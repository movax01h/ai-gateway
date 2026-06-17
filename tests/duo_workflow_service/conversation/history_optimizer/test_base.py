"""Tests for the ``HistoryOptimizer`` abstract base class."""

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from duo_workflow_service.conversation.history_optimizer.base import HistoryOptimizer
from duo_workflow_service.conversation.history_optimizer.schema import (
    OptimizationResult,
)


class TestHistoryOptimizer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            # pylint: disable=abstract-class-instantiated
            HistoryOptimizer()

    @pytest.mark.asyncio
    async def test_concrete_subclass_works(self):
        class NoopOptimizer(HistoryOptimizer):
            async def optimize(self, history: list[BaseMessage]) -> OptimizationResult:
                return OptimizationResult(messages=history, was_modified=False)

        opt = NoopOptimizer()
        history = [HumanMessage(content="x")]
        result = await opt.optimize(history)
        assert result.was_modified is False
        assert result.messages is history
