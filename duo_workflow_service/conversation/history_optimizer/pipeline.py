"""Pipeline that composes multiple ``HistoryOptimizer`` instances in order."""

from langchain_core.messages import BaseMessage

from duo_workflow_service.conversation.history_optimizer.base import HistoryOptimizer
from duo_workflow_service.conversation.history_optimizer.schema import (
    OptimizationResult,
)


class HistoryOptimizerPipeline:
    """Ordered pipeline of history optimizers.

    The pipeline calls each optimizer's ``optimize()`` method in order,
    threading the (possibly-modified) message list through. Each optimizer's
    result is collected; callers decide how to use them (e.g., extract a
    specific result, merge ``ui_chat_logs`` into state).
    """

    def __init__(self, optimizers: list[HistoryOptimizer]):
        self._optimizers = optimizers

    @property
    def optimizers(self) -> list[HistoryOptimizer]:
        """Return the underlying ordered list of optimizers."""
        return self._optimizers

    async def optimize(
        self,
        history: list[BaseMessage],
    ) -> tuple[list[BaseMessage], list[OptimizationResult]]:
        """Run all optimizers in order.

        When an optimizer reports ``was_modified=True``, its output replaces
        the running history before the next optimizer is invoked. Results are
        returned in the same order as ``self.optimizers``.

        Args:
            history: Initial conversation history.

        Returns:
            A tuple ``(messages, results)``. ``messages`` is the final
            history after running every optimizer. ``results`` is the list of
            per-optimizer outcomes (one per optimizer, in order).
        """
        results: list[OptimizationResult] = []
        for opt in self._optimizers:
            result = await opt.optimize(history)
            if result.was_modified:
                history = result.messages
            results.append(result)
        return history, results
