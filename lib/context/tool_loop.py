"""Repeated-tool-call tracking for a workflow session.

Measures how much of a session's tool activity is repetition, so we can size the
"agent stuck in a tool loop" failure mode in production before building detection
for it. Purely observational: nothing here changes agent behaviour.

Two consecutive-repeat metrics are tracked deliberately. Keying on
``(tool, args)`` is what a naive detector would trip on; keying on
``(tool, args, result)`` additionally requires the result to be unchanged. Same
call with a *changing* result is legitimate (polling a pipeline, retrying a
transient failure), so the difference between the two metrics is the
false-positive rate a ``(tool, args)``-only detector would incur.

Calls are identified by hash, never retained verbatim: tool arguments and
results must not reach an analytics payload, and some are large (a whole file
body for ``create_file_with_contents``), so holding a few hundred for the
session would cost real memory. ``compute_response_hash`` is reused for this
because it already falls back to ``str()`` when JSON serialisation fails, which
model- and API-shaped payloads do more often than one would like -- a dict with
mixed-type keys cannot be sorted, and ``json.dumps(default=...)`` covers values
only, never keys. Recording happens after the tools have run, so raising here
would discard work that already succeeded.

See gitlab-org/modelops/applied-ml/code-suggestions/ai-assist#2777.
"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

from duo_workflow_service.security.security_utils import compute_response_hash

__all__ = [
    "ToolLoopStats",
    "build_tool_loop_session_summary_extras",
    "init_tool_loop_counters",
    "record_tool_calls",
]


@dataclass
class ToolLoopStats:
    """Running repeated-tool-call counters for one workflow session.

    Mutated in place rather than rebound, which is what lets the counters
    survive ``contextvars.copy_context()``: a copied context maps the same
    ``ToolLoopStats`` object, so a nested task's updates are visible to the
    parent. (Contrast the plain-int counters in ``lib.context.orbit``, which
    have to be incremented in the parent task's own context.)
    """

    total_calls: int = 0
    max_consecutive_args: int = 0
    max_consecutive_args_result: int = 0

    _unique_args: set[str] = field(default_factory=set)
    _current_args: Optional[str] = None
    _current_args_streak: int = 0
    _current_args_result: Optional[str] = None
    _current_args_result_streak: int = 0

    @property
    def unique_calls(self) -> int:
        """Number of distinct ``(tool, args)`` pairs seen this session."""
        return len(self._unique_args)

    def record(self, tool_name: str, args: Any, result: Any) -> None:
        """Fold one executed tool call into the counters."""
        self.total_calls += 1

        args_fp = compute_response_hash([tool_name, args])
        self._unique_args.add(args_fp)
        self._current_args_streak = (
            self._current_args_streak + 1 if args_fp == self._current_args else 1
        )
        self._current_args = args_fp
        self.max_consecutive_args = max(
            self.max_consecutive_args, self._current_args_streak
        )

        args_result_fp = compute_response_hash([tool_name, args, result])
        self._current_args_result_streak = (
            self._current_args_result_streak + 1
            if args_result_fp == self._current_args_result
            else 1
        )
        self._current_args_result = args_result_fp
        self.max_consecutive_args_result = max(
            self.max_consecutive_args_result, self._current_args_result_streak
        )


tool_loop_stats: ContextVar[Optional[ToolLoopStats]] = ContextVar(
    "tool_loop_stats", default=None
)


def init_tool_loop_counters() -> None:
    """Start fresh counters for a workflow session.

    Also guards against ``ContextVar`` leakage should sessions ever run
    sequentially in one context, the same way ``init_orbit_counters`` does.
    """
    tool_loop_stats.set(ToolLoopStats())


def record_tool_calls(calls: list[tuple[str, Any, Any]]) -> None:
    """Record executed ``(tool_name, args, result)`` triples, in execution order.

    A no-op when the counters were never initialised (e.g. offline mode, which
    returns before ``init_tool_loop_counters``), so callers do not have to care.
    """
    stats = tool_loop_stats.get()
    if stats is None:
        return
    for tool_name, args, result in calls:
        stats.record(tool_name, args, result)


def build_tool_loop_session_summary_extras(
    workflow_id: str, workflow_type: str
) -> Optional[dict]:
    """Build kwargs for ``WORKFLOW_TOOL_LOOP_SESSION_SUMMARY``.

    Returns ``None`` when no tools ran, so sessions with nothing to report emit
    nothing at all.
    """
    stats = tool_loop_stats.get()
    if stats is None or stats.total_calls == 0:
        return None
    return {
        "value": workflow_id,
        "workflow_type": workflow_type,
        "total_tool_calls": stats.total_calls,
        "unique_tool_calls": stats.unique_calls,
        "max_consecutive_identical_calls": stats.max_consecutive_args,
        "max_consecutive_identical_with_result": stats.max_consecutive_args_result,
    }
