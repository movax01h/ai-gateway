"""Compatibility shim for the legacy ``maybe_compact_history`` dispatcher.

This module preserves the legacy entry point used by call sites that have
not yet been migrated to ``HistoryOptimizerPipeline``. The shim constructs a
transient pipeline from the legacy arguments and returns a result tuple in
the same shape as before.

Deprecated: callers should migrate to ``build_history_optimizer_pipeline``.
This module is deleted in MR 3 once all call sites have been migrated.
"""

from dependency_injector.wiring import Provide, inject
from langchain_core.messages import BaseMessage

from ai_gateway.container import ContainerApplication
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    CompactionOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.optimizers.legacy_trim import (
    LegacyTrimOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionResult,
)
from lib.internal_events.client import InternalEventsClient


@inject
async def maybe_compact_history(
    compactor: CompactionOptimizer | None,
    history: list[BaseMessage],
    agent_name: str,
    internal_events_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
) -> tuple[list[BaseMessage], CompactionResult | None]:
    """Compact or trim conversation history (legacy compatibility shim).

    When ``compactor`` is provided, runs a single-stage pipeline with that
    compactor. When ``compactor`` is ``None``, falls back to a single-stage
    pipeline running ``LegacyTrimOptimizer``.

    Returns ``(messages, result)``. ``result`` is the ``CompactionResult``
    produced by the compactor (success or no-op) when one ran, or ``None``
    when the legacy trim path ran. This matches the pre-refactor contract
    exactly so existing call sites work unchanged.
    """
    if compactor is not None:
        pipeline = HistoryOptimizerPipeline([compactor])
    else:
        pipeline = HistoryOptimizerPipeline(
            [
                LegacyTrimOptimizer(
                    agent_name=agent_name,
                    internal_events_client=internal_events_client,
                )
            ]
        )

    messages, results = await pipeline.optimize(history)
    compaction_result = next(
        (r for r in results if isinstance(r, CompactionResult)), None
    )
    return messages, compaction_result
