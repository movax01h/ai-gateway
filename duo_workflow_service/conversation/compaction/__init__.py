"""Compatibility re-exports for the legacy compaction package.

The canonical implementation lives under
``duo_workflow_service.conversation.history_optimizer``. This package keeps
existing imports working while callers migrate.

Note: ``CompactionOptimizer`` is intentionally not re-exported here. New code
should import it directly from
``duo_workflow_service.conversation.history_optimizer``. Re-exporting it
through this package's ``__init__.py`` would create a circular import: the
new optimizer module imports ``compaction.utils``, which forces this
``__init__.py`` to fully execute before the optimizer module finishes
initializing.
"""

from duo_workflow_service.conversation.compaction.compactor import (
    ConversationCompactor,
    create_conversation_compactor,
)
from duo_workflow_service.conversation.compaction.integration import (
    maybe_compact_history,
)
from duo_workflow_service.conversation.compaction.schema import (
    CompactionConfig,
    CompactionResult,
    MessageSlices,
)
from duo_workflow_service.conversation.compaction.utils import (
    is_turn_complete,
    resolve_recent_messages_internal,
    slice_for_summarization,
)

__all__ = [
    "ConversationCompactor",
    "create_conversation_compactor",
    "CompactionConfig",
    "CompactionResult",
    "MessageSlices",
    "is_turn_complete",
    "resolve_recent_messages_internal",
    "slice_for_summarization",
    "maybe_compact_history",
]
