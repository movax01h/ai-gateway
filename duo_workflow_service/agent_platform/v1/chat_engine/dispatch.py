from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Optional, Self

from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.errors.typing import InvalidRequestException

__all__ = [
    "LEGACY_SESSION_MESSAGE",
    "EntryDispatch",
    "EntryEvent",
    "EntryEventKind",
    "EntryMode",
    "EntryRejectedException",
    "EntryTip",
    "TipState",
    "classify_entry",
    "classify_tip",
]

LEGACY_SESSION_MESSAGE = (
    "This chat was started on the previous chat engine and cannot continue "
    "here. Start a new chat."
)


class EntryRejectedException(InvalidRequestException):
    """A session entry the dispatch table rejects.

    Raised after the graph is compiled, so the checkpointer has already sent Rails the entry's status event. The
    ``InvalidRequestException`` base is what ``GitLabWorkflow.__aexit__`` handles without a ``drop``: it reconciles
    Rails back to the status the tip declares, and the server answers ``INVALID_ARGUMENT`` with the reason. The
    session is left as it was and takes its next entry.
    """


class EntryEventKind(StrEnum):
    """What the client sent with the session entry."""

    MESSAGE = "message"
    DECISION = "decision"
    RECONNECT = "reconnect"


@dataclass(frozen=True)
class EntryEvent:
    """The client event of a session entry, as the dispatch table reads it."""

    kind: EntryEventKind
    pinned: bool = False

    @classmethod
    def from_request(
        cls, *, has_user_content: bool, decision: Optional[str], pinned: bool
    ) -> Self:
        """A decision outranks a message, and an entry that carries neither is a reconnect.

        The caller decides what counts as user content: text, and anything else a message can carry without it.
        """
        if decision:
            kind = EntryEventKind.DECISION
        elif has_user_content:
            kind = EntryEventKind.MESSAGE
        else:
            kind = EntryEventKind.RECONNECT
        return cls(kind=kind, pinned=pinned)


@dataclass(frozen=True)
class EntryTip:
    """What classification reads from the checkpoint an entry continues from.

    ``next_nodes`` is structural: LangGraph derives it from the compiled graph, so it holds on the entry path where
    the checkpoint carries no pending writes. ``channels`` are the raw channel names the checkpoint stores, which a
    state snapshot filters down to the graph's own schema. ``pre_run_input`` marks LangGraph's input checkpoint,
    written before the first node of a run.
    """

    next_nodes: tuple[str, ...]
    channels: frozenset[str]
    pre_run_input: bool


class TipState(StrEnum):
    """Where the session is, as far as the chat engine needs to know.

    WAITING and RUNNING are both inside a turn. What is waiting, and on which node, is ``Flow``'s to handle.
    """

    NONE = "none"
    BOUNDARY = "boundary"
    WAITING = "waiting"
    RUNNING = "running"
    LEGACY = "legacy"


class EntryMode(StrEnum):
    """How a session entry is classified: the ``Mode`` column of the dispatch table.

    RESUME and RETRY hand the entry to ``Flow``'s own resume and retry paths.
    """

    START = "start"
    TURN = "turn"
    RESUME = "resume"
    RETRY = "retry"
    REJECT = "reject"


@dataclass(frozen=True)
class EntryDispatch:
    """Classification result: the mode, the tip state it was read from, and a reason when rejected."""

    mode: EntryMode
    tip_state: TipState
    reason: Optional[str] = None


# Channels only legacy ``chat.Workflow`` declares. Its first step writes every
# required one, None included, and a ``FlowState`` checkpoint stores none. So the
# check needs no LangGraph channel names, and ``FlowState`` can gain a channel
# without turning older sessions legacy.
_LEGACY_CHAT_CHANNELS: Final = frozenset(ChatWorkflowState.__annotations__) - frozenset(
    FlowState.__annotations__
)


def classify_tip(
    tip: Optional[EntryTip], *, waiting_on_user: bool, rolled_back: bool = False
) -> TipState:
    """Place the session from the checkpoint an entry continues from.

    ``waiting_on_user`` is ``Flow``'s own signal that the turn waits on the user: the entry arrived as a RESUME,
    which Rails sends from its awaiting statuses. The tip only says whether a turn is in progress.

    ``rolled_back`` says the entry continues from where ``Flow``'s stop recovery landed rather than from the tip. The
    walk lands on the newest ``INPUT_REQUIRED`` checkpoint, and on the engine that is the stopped turn's input
    checkpoint, which holds the boundary its run started from.
    """
    if tip is None:
        return TipState.NONE

    if tip.channels & _LEGACY_CHAT_CHANNELS:
        return TipState.LEGACY

    if tip.pre_run_input:
        # The input checkpoint still carries the previous turn's state, and on
        # the entry path its queued input is not visible, so it reads like the
        # boundary. At the tip its run is in progress, and nothing in it can be
        # waiting. Rolled back to, it is that boundary.
        return TipState.BOUNDARY if rolled_back else TipState.RUNNING

    if tip.next_nodes:
        return TipState.WAITING if waiting_on_user else TipState.RUNNING

    return TipState.BOUNDARY


_NO_CHECKPOINTS = "The session has no checkpoints yet; send a message to start it."
_STALE_DECISION = "The session is not waiting on an approval; the decision is stale."
_AWAITING_MESSAGE = "The session is waiting for your next message."
_PIN_NEEDS_MESSAGE = (
    "A pinned checkpoint takes a new message, not an approval or a reconnect."
)
_RETRY_FIRST = (
    "The last turn hasn't finished. Retry it before sending a new message or decision."
)

# The dispatch table of ``docs/flow_registry/chat_engine.md``, "Where a turn
# begins": tip state and client event in, mode out. A string is a REJECT with
# the reason the client sees. Every pair has a row.
_DISPATCH_TABLE: Final[dict[tuple[TipState, EntryEventKind], EntryMode | str]] = {
    (TipState.NONE, EntryEventKind.MESSAGE): EntryMode.START,
    (TipState.NONE, EntryEventKind.DECISION): _NO_CHECKPOINTS,
    (TipState.NONE, EntryEventKind.RECONNECT): _NO_CHECKPOINTS,
    (TipState.BOUNDARY, EntryEventKind.MESSAGE): EntryMode.TURN,
    (TipState.BOUNDARY, EntryEventKind.DECISION): _STALE_DECISION,
    (TipState.BOUNDARY, EntryEventKind.RECONNECT): _AWAITING_MESSAGE,
    (TipState.WAITING, EntryEventKind.MESSAGE): EntryMode.RESUME,
    (TipState.WAITING, EntryEventKind.DECISION): EntryMode.RESUME,
    (TipState.WAITING, EntryEventKind.RECONNECT): EntryMode.RETRY,
    (TipState.RUNNING, EntryEventKind.MESSAGE): _RETRY_FIRST,
    (TipState.RUNNING, EntryEventKind.DECISION): _RETRY_FIRST,
    (TipState.RUNNING, EntryEventKind.RECONNECT): EntryMode.RETRY,
    (TipState.LEGACY, EntryEventKind.MESSAGE): LEGACY_SESSION_MESSAGE,
    (TipState.LEGACY, EntryEventKind.DECISION): LEGACY_SESSION_MESSAGE,
    (TipState.LEGACY, EntryEventKind.RECONNECT): LEGACY_SESSION_MESSAGE,
}


def classify_entry(
    tip: Optional[EntryTip],
    event: EntryEvent,
    *,
    waiting_on_user: bool,
    rolled_back: bool = False,
) -> EntryDispatch:
    """Classify a session entry from the checkpoint it continues from and the client event.

    A message or a decision on a turn in progress is rejected with a reason that asks for a retry first: the message
    isn't dropped, and the decision has nothing paused to answer. A retry is an entry with no message, which replays the
    turn through ``Flow``'s retry.

    A pinned checkpoint is never waiting, because nothing is live there. It takes a message only, as a TURN that forks
    from it: a decision or a reconnect on it is rejected. A legacy session is rejected as legacy, pinned or not. A
    rollback goes through the table from where the walk landed.
    """
    state = classify_tip(
        tip,
        waiting_on_user=waiting_on_user and not event.pinned,
        rolled_back=rolled_back,
    )
    outcome: EntryMode | str

    if event.pinned and state is TipState.NONE:
        outcome = _NO_CHECKPOINTS
    elif event.pinned and state is not TipState.LEGACY:
        outcome = (
            EntryMode.TURN
            if event.kind is EntryEventKind.MESSAGE
            else _PIN_NEEDS_MESSAGE
        )
    else:
        outcome = _DISPATCH_TABLE[state, event.kind]

    if isinstance(outcome, EntryMode):
        return EntryDispatch(outcome, state)
    return EntryDispatch(EntryMode.REJECT, state, outcome)
