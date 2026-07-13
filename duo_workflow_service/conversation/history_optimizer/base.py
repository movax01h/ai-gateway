"""Abstract base class for history optimizers."""

from abc import ABC, abstractmethod

from langchain_core.messages import BaseMessage

from duo_workflow_service.conversation.history_optimizer.schema import (
    OptimizationResult,
)


class HistoryOptimizer(ABC):
    """Optimizes conversation history before an LLM invocation.

    Implementations self-guard: ``optimize()`` is always called by the
    pipeline; the optimizer returns
    ``OptimizationResult(was_modified=False, messages=history)`` when no action
    is needed (e.g., history under threshold). There is intentionally no
    ``should_run()`` method on the interface.
    """

    @abstractmethod
    async def optimize(self, history: list[BaseMessage]) -> OptimizationResult:
        """Optimize the given conversation history.

        Args:
            history: Current conversation history.

        Returns:
            An ``OptimizationResult`` (or subclass) describing the outcome.
            ``was_modified`` indicates whether ``messages`` differs from the
            input. ``ui_chat_logs`` carries UI artifacts to surface.
        """
