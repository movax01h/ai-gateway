from enum import StrEnum
from typing import Callable, Optional, TypedDict

from langchain_core.messages import ToolCall

from duo_workflow_service.agent_platform.v1.state import IOKey

__all__ = [
    "DELEGATION_CALL_ID_CONTEXT_KEY",
    "SUBSESSION_RUNS_CONTEXT_KEY",
    "DelegationFatalError",
    "DelegationStatus",
    "SubsessionRun",
    "SubsessionRunKeyFactory",
    "format_delegation_result",
    "require_tool_call_id",
]

# Factory that builds the IOKey holding one dispatched delegation's run record,
# given the ``delegate_task`` call ID that dispatched it.
SubsessionRunKeyFactory = Callable[[str], IOKey]

# Context key (scoped to the supervisor's own name -- see
# ``SupervisorAgentComponentV2._subsession_run_key_factory``) holding the
# ``SubsessionRun`` records of dispatched delegations, keyed by the
# ``delegate_task`` call ID that dispatched each one.
SUBSESSION_RUNS_CONTEXT_KEY = "subsession_runs"

# Context key seeded by ``DelegationPrepareNode`` on every dispatched initial
# state, and read back by ``SubagentDispatchNode`` to write that dispatch's run
# record under the call ID it belongs to.
DELEGATION_CALL_ID_CONTEXT_KEY = "_delegation_call_id"


def require_tool_call_id(tc: ToolCall) -> str:
    """Return ``tc["id"]``, raising if it's missing.

    ``ToolCall.id`` is typed as ``Optional[str]`` in langchain-core, but is
    always populated by the model provider for a genuine tool call; a missing
    id here means the message layer upstream is broken (a wiring bug, not a
    recoverable LLM mistake), and we cannot build a ``ToolMessage`` response
    for this call without it -- nor address the run record it dispatched.

    Raises:
        DelegationFatalError: If ``tc["id"]`` is ``None``.
    """
    call_id = tc["id"]
    if call_id is None:
        raise DelegationFatalError(
            f"Tool call {tc['name']!r} is missing an id; cannot build a "
            f"ToolMessage response for it."
        )
    return call_id


class DelegationStatus(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"


class DelegationFatalError(Exception):
    """Unrecoverable delegation error indicating a graph wiring or state corruption bug.

    These errors propagate up and stop execution — they should never occur during normal operation and cannot be
    meaningfully handled by the LLM.
    """


class SubsessionRun(TypedDict):
    """One dispatched delegation's complete outcome, as one atomically-written record.

    Written by ``SubagentDispatchNode`` — the node that owns a synchronous
    subagent's whole runtime — under
    ``SUBSESSION_RUNS_CONTEXT_KEY[call_id]``, and read back by
    ``DelegationCollectNode`` to answer that ``delegate_task`` call. Keying by
    call ID (rather than by subsession, as the outcome once was) is what makes
    the record self-contained: the writer of a call's outcome and the reader of
    it agree on one slot, no turn-scoped manifest has to survive in between,
    and a resumed subsession's second dispatch can never be mistaken for its
    first.

    Every field is written on every dispatch, including the ``None`` ones. The
    ``context`` reducer deep-merges, so a field left out of a re-run of the
    same call (a replayed ``Send`` after an interrupt) would keep the previous
    run's value: a stale ``error`` would mask a successful retry, and a stale
    ``final_answer`` would report a retry that produced no answer as having
    completed with the earlier answer.

    Attributes:
        subsession_id: The subsession this call ran, needed to
            report the ID back to the supervisor and to label the UI entry.
        status: ``ERROR`` if the dispatch itself failed (see ``error``),
            otherwise ``COMPLETED`` — the subagent's graph ran to its own
            terminal node. Whether it actually *answered* is
            ``final_answer``'s business.
        error: The failure message when ``status`` is ``ERROR``, else ``None``.
        final_answer: The subagent's answer, or ``None`` when it produced none
            (or the dispatch failed).
    """

    subsession_id: int
    status: DelegationStatus
    error: Optional[str]
    final_answer: Optional[str]


def format_delegation_result(
    subagent_name: str,
    status: DelegationStatus,
    content: str,
) -> str:
    """Format a subagent's delegation outcome as XML for the supervisor.

    The XML separates which subagent answered, and whether it succeeded, from the answer itself, so the supervisor's LLM
    can tell a failed delegation from a successful one without parsing prose.

    The subsession ID is deliberately absent. It used to be reported so the LLM could name a subsession to resume; now
    that nothing can be resumed, showing an identifier no argument accepts only invites the model to try. It is still
    recorded for UI attribution and structured logs, which is where it is actually read.
    """
    if status == DelegationStatus.ERROR:
        return (
            f"<delegation_result>\n"
            f"  <subagent_name>{subagent_name}</subagent_name>\n"
            f"  <status>{status}</status>\n"
            f"  <error>\n"
            f"    {content}\n"
            f"  </error>\n"
            f"</delegation_result>"
        )

    return (
        f"<delegation_result>\n"
        f"  <subagent_name>{subagent_name}</subagent_name>\n"
        f"  <status>{status}</status>\n"
        f"  <result>\n"
        f"    {content}\n"
        f"  </result>\n"
        f"</delegation_result>"
    )
