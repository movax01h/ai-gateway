"""Backwards-compatible re-exports for the conversation compactor.

The canonical home is
``duo_workflow_service.conversation.history_optimizer.optimizers.compaction``.
``ConversationCompactor`` is a legacy alias for ``CompactionOptimizer``.

This module also re-exports module-level symbols (constants, log,
``get_model_metadata``, ``duo_workflow_metrics``) so that existing tests that
patch them via the legacy import path keep working until they migrate.
"""

# pylint: disable=unused-import
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (  # noqa: F401
    COMPACTION_CONTINUE_MESSAGE,
    COMPACTION_PROMPT_ID,
    COMPACTION_PROMPT_MANUAL_ID,
    COMPACTION_PROMPT_VERSION,
    CompactionOptimizer,
    CompactionStatus,
    create_conversation_compactor,
    duo_workflow_metrics,
    get_current_model_max_context_token_limit,
    get_model_metadata,
    log,
)

# Legacy alias: the class was renamed but external callers and tests still
# refer to it by its original name.
ConversationCompactor = CompactionOptimizer

__all__ = [
    "COMPACTION_CONTINUE_MESSAGE",
    "COMPACTION_PROMPT_ID",
    "COMPACTION_PROMPT_MANUAL_ID",
    "COMPACTION_PROMPT_VERSION",
    "CompactionOptimizer",
    "ConversationCompactor",
    "CompactionStatus",
    "create_conversation_compactor",
]
