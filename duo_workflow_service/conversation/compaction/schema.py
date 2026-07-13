"""Backwards-compatible re-exports for compaction schema types.

The canonical home for these types is
``duo_workflow_service.conversation.history_optimizer.schema``. This module
re-exports them so existing imports continue to work while callers migrate.
"""

from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    CompactionResult,
    MessageSlices,
)

__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "MessageSlices",
]
