"""Helpers for computing the ``cancelled_turn`` context envelope.

This module is extracted from ``flow/base.py`` and owns the logic that
derives the human-visible transcript of a cancelled workflow turn so that
``Flow._resolve_stop_recovery`` can surface it as a built-in context
envelope for the model.

Public entry point
------------------
``cancelled_turn_context`` — given the decoded tip and boundary
``Checkpoint`` dicts (langgraph ``channel_values`` containers), extracts
their ``ui_chat_log`` channel values, computes the delta, and returns the
filtered list of :class:`~duo_workflow_service.entities.state.UiChatLog`
entries that were discarded by the rollback.

Internal helpers
----------------
``_extract_ui_chat_log`` — extracts the ``ui_chat_log`` channel value from a
``CheckpointTuple``, tolerating a missing tuple or channel.

``_compute_cancelled_turn_delta`` — pure function that computes the delta
between two ``ui_chat_log`` lists, applies type filtering, strips
``tool_info`` payloads, and enforces entry-count and byte-size caps.
"""

import json
from typing import Any, Optional, cast

from langgraph.checkpoint.base import CheckpointTuple

from duo_workflow_service.entities.state import MessageTypeEnum, UiChatLog

__all__ = ["CANCELLED_TURN_CATEGORY", "cancelled_turn_context"]

# Engine-level built-in category for the cancelled-turn context envelope.
# Handled like executor-context categories (no flow-config declaration required).
CANCELLED_TURN_CATEGORY = "cancelled_turn"

# Size caps for the cancelled-turn ui_chat_log delta to guard against
# pathological sessions.  The common case is one user message plus a short
# partial agent reply, so these limits are generous.
_CANCELLED_TURN_MAX_ENTRIES = 20
_CANCELLED_TURN_MAX_BYTES = 32 * 1024  # 32 KiB

# Only surface human-visible message types in the cancelled-turn context.
# Tool messages and workflow-end markers are internal and not referenced by
# users in follow-up messages.
_CANCELLED_TURN_KEPT_TYPES = frozenset({MessageTypeEnum.USER, MessageTypeEnum.AGENT})


def _compute_cancelled_turn_delta(
    *,
    tip: list[UiChatLog],
    boundary: list[UiChatLog],
    log: Any,
) -> list[UiChatLog]:
    """Return the ui_chat_log entries that were discarded by the rollback.

    The delta is the tail of *tip* beyond the length of *boundary*, relying on
    the append-only prefix assumption that ``_list_delta`` in the checkpointer
    already depends on.  If the prefix assumption is violated (e.g. a compaction
    landed between boundary and tip), the function degrades gracefully by
    returning an empty list rather than surfacing a potentially misleading diff.

    The result is filtered to human-visible message types only (USER and AGENT),
    with ``tool_info`` payloads stripped to keep the envelope compact.  Entry
    count and byte size are capped to guard against pathological sessions.

    Args:
        tip: The ``ui_chat_log`` from the newest (pre-rollback) checkpoint.
        boundary: The ``ui_chat_log`` from the boundary checkpoint.
        log: A structlog logger used to emit a warning when the prefix
            assumption is violated.

    Returns:
        A (possibly empty) list of ``UiChatLog`` entries representing the
        cancelled exchange, ready to be serialised into a ``cancelled_turn``
        context envelope.
    """
    boundary_len = len(boundary)
    if boundary_len > len(tip) or tip[:boundary_len] != boundary:
        log.warning(
            "cancelled_turn delta: prefix assumption violated; degrading to empty delta",
            tip_len=len(tip),
            boundary_len=boundary_len,
        )
        return []

    raw_delta = tip[boundary_len:]

    filtered: list[UiChatLog] = []
    for entry in raw_delta:
        if entry.get("message_type") not in _CANCELLED_TURN_KEPT_TYPES:
            continue
        # Shallow-copy and strip tool_info to keep the envelope compact.
        stripped: UiChatLog = cast(UiChatLog, {**entry, "tool_info": None})
        filtered.append(stripped)

    if len(filtered) > _CANCELLED_TURN_MAX_ENTRIES:
        log.warning(
            "cancelled_turn delta exceeds entry cap; truncating",
            original_count=len(filtered),
            cap=_CANCELLED_TURN_MAX_ENTRIES,
        )
        filtered = filtered[:_CANCELLED_TURN_MAX_ENTRIES]

    # Approximate byte size by serialising to JSON; drop the whole delta rather
    # than silently truncating mid-entry to avoid partial context.
    try:
        serialised = json.dumps(filtered, default=str)
        encoded = serialised.encode()
        if len(encoded) > _CANCELLED_TURN_MAX_BYTES:
            log.warning(
                "cancelled_turn delta exceeds byte cap; dropping to avoid oversized envelope",
                byte_size=len(encoded),
                cap=_CANCELLED_TURN_MAX_BYTES,
            )
            return []
    except (TypeError, ValueError) as exc:
        log.warning(
            "cancelled_turn delta serialisation failed; dropping",
            exc_info=exc,
        )
        return []

    return filtered


def _extract_ui_chat_log(
    checkpoint_tuple: Optional[CheckpointTuple],
) -> list[UiChatLog]:
    """Extract the ``ui_chat_log`` channel value from a ``CheckpointTuple``.

    Args:
        checkpoint_tuple: A langgraph ``CheckpointTuple``, or ``None``.

    Returns:
        The ``ui_chat_log`` list from the checkpoint, or an empty list when
        *checkpoint_tuple* is ``None`` or carries no ``ui_chat_log``.
    """
    if not checkpoint_tuple:
        return []
    return (
        checkpoint_tuple.checkpoint.get("channel_values", {}).get("ui_chat_log", [])
        or []
    )


def cancelled_turn_context(
    *,
    latest: Optional[CheckpointTuple],
    boundary: Optional[CheckpointTuple],
    log: Any,
) -> list[UiChatLog]:
    """Compute the cancelled-turn ``ui_chat_log`` delta for a stop-recovery rollback.

    Extracts the ``ui_chat_log`` channel value from each ``CheckpointTuple``
    and delegates to :func:`_compute_cancelled_turn_delta` to produce the filtered
    list of entries that were discarded by the rollback.

    Args:
        latest: The ``CheckpointTuple`` for the pre-rollback tip (the newest
            checkpoint before the stop), or ``None`` when no tip checkpoint is
            available.
        boundary: The ``CheckpointTuple`` for the stable conversational boundary
            the flow is rolling back to, or ``None`` when the flow was stopped
            before its first pause point.
        log: A structlog logger used to emit warnings on delta-computation
            anomalies.

    Returns:
        A (possibly empty) list of :class:`~duo_workflow_service.entities.state.UiChatLog`
        entries representing the human-visible transcript of the cancelled turn.
    """
    tip_ui_chat_log = _extract_ui_chat_log(latest)
    boundary_ui_chat_log = _extract_ui_chat_log(boundary)

    return _compute_cancelled_turn_delta(
        tip=tip_ui_chat_log,
        boundary=boundary_ui_chat_log,
        log=log,
    )
