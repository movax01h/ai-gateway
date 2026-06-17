"""Backwards-compatible re-exports for compaction helpers.

The canonical home for these helpers is
``duo_workflow_service.conversation.history_optimizer.optimizers._compaction_utils``.
This module re-exports them so existing imports continue to work while
callers migrate.
"""

from duo_workflow_service.conversation.history_optimizer.optimizers._compaction_utils import (
    is_turn_complete,
    resolve_recent_messages_internal,
    slice_for_summarization,
    strip_tool_metadata,
)

__all__ = [
    "is_turn_complete",
    "resolve_recent_messages_internal",
    "slice_for_summarization",
    "strip_tool_metadata",
]
