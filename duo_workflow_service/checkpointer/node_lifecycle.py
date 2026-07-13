"""Runtime node-lifecycle event log for live flow visualization.

A flow's graph nodes compile from design-time components (``researcher#agent``
and ``researcher#tools`` both belong to the component ``researcher``). As a flow
runs we surface an append-only log of node-lifecycle *events* — each node run
starting, ending, or erroring — so the client can reconstruct the live
execution path by folding the log: which components are currently executing,
which have run, how many times a node was entered (loops), and which errored.
Crucially this also captures *silent* nodes (routers, deterministic steps) that
emit no chat output, which a chat-log-derived view cannot see.

This module holds the log and the callback handler that feeds it. The log
(:class:`NodeEvent` / :class:`NodeEventLog`) is pure in-memory state with no
LangGraph/LangChain dependency, so it is trivial to test and reason about; the
handler is the thin adapter onto LangChain callbacks.

Concurrency: the handler's callbacks fire on the single asyncio event loop that
runs the graph (LangGraph awaits async callbacks inline). None of the mutating
methods ``await`` mid-operation, so each runs atomically with respect to the
loop — parallel branches and the agent ReAct loop interleave only at event
boundaries, never inside a record/append. No locking is required.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.errors import GraphInterrupt

from duo_workflow_service.agent_platform.node_naming import component_name_from_node


class NodePhase(str, Enum):
    """Lifecycle transition of a single node run."""

    STARTED = "started"
    ENDED = "ended"
    ERRORED = "errored"


@dataclass(frozen=True)
class NodeEvent:
    """One node-lifecycle transition; the unit clients fold into canvas state.

    ``run_id`` is the LangGraph run id of the node execution. It pairs a
    ``STARTED`` with its terminal event and distinguishes concurrent or repeated
    runs of the same component — parallel branches, agent ReAct loop iterations,
    and a node re-run after a human-in-the-loop resume (which gets a fresh run
    id). ``component`` is the design-time component the client draws on the
    canvas, recovered from the runtime node name.
    """

    run_id: str
    component: str
    phase: NodePhase

    def to_dict(self) -> dict[str, str]:
        """Wire form embedded in the checkpoint's ``channel_values.node_events``."""
        return {
            "run_id": self.run_id,
            "component": self.component,
            "phase": self.phase.value,
        }


class NodeEventLog:
    """Append-only log of node-lifecycle events for a single flow run.

    Pure state: ``record`` appends, ``events`` exposes the ordered log. Deriving
    active / visited / loop-count / errored from the log is the client's job, so
    this class stays free of any view policy.
    """

    def __init__(self) -> None:
        self._events: list[NodeEvent] = []

    def record(self, run_id: str, component: str, phase: NodePhase) -> None:
        self._events.append(NodeEvent(run_id=run_id, component=component, phase=phase))

    @property
    def events(self) -> list[NodeEvent]:
        return self._events


# LangGraph tags the top-level run of each graph node with ``graph:step:<n>``.
# Nested runnables inside a node carry ``seq:step:<n>`` instead, so this prefix
# reliably distinguishes a node boundary from inner-runnable noise.
_NODE_BOUNDARY_TAG_PREFIX = "graph:step:"


def _is_node_boundary(tags: Optional[list[str]]) -> bool:
    return any(tag.startswith(_NODE_BOUNDARY_TAG_PREFIX) for tag in (tags or []))


class NodeLifecycleCallbackHandler(AsyncCallbackHandler):  # pylint: disable=too-many-ancestors
    """Feeds a :class:`NodeEventLog` from LangGraph node-execution callbacks.

    LangChain fires ``on_chain_start``/``on_chain_end``/``on_chain_error`` for
    every runnable, not just graph nodes, and nested runnables inherit their
    parent node's ``langgraph_node`` metadata — so that field alone over-counts.
    Node boundaries are identified by the ``graph:step:`` tag. The handler keeps
    a ``run_id -> component`` map of currently-open node runs so it records a
    terminal event only for runs it actually opened (never an orphan terminal
    for an inner runnable or the graph root) and can attach the component to
    that terminal event.

    A node that pauses for human input raises ``GraphInterrupt``, which surfaces
    via ``on_chain_error``. That is a *pause*, not a failure — the run genuinely
    ends and resuming re-runs the node under a new run id — so it is recorded as
    ``ENDED``. Only non-interrupt errors are recorded as ``ERRORED``.
    """

    def __init__(self, event_log: NodeEventLog) -> None:
        self._event_log = event_log
        # run_id -> component, for node runs that have started but not terminated.
        self._open_runs: dict[str, str] = {}

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not _is_node_boundary(tags):
            return
        component = component_name_from_node((metadata or {}).get("langgraph_node"))
        if not component:
            return
        key = str(run_id)
        self._open_runs[key] = component
        self._event_log.record(key, component, NodePhase.STARTED)

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._terminate(str(run_id), NodePhase.ENDED)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # A node pausing for human input raises GraphInterrupt through
        # on_chain_error; that is a clean yield, not a failure.
        phase = (
            NodePhase.ENDED if isinstance(error, GraphInterrupt) else NodePhase.ERRORED
        )
        self._terminate(str(run_id), phase)

    def _terminate(self, run_id: str, phase: NodePhase) -> None:
        # No-op for run ids we never opened: inner runnables, the graph root, or
        # a run already terminated.
        component = self._open_runs.pop(run_id, None)
        if component is None:
            return
        self._event_log.record(run_id, component, phase)
