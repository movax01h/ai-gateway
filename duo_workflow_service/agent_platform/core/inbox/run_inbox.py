import asyncio
from dataclasses import dataclass, field
from typing import NewType, Optional

from langgraph.runtime import Runtime
from langgraph.typing import ContextT

from duo_workflow_service.agent_platform.core.inbox.message import (
    Answer,
    InboundMessage,
)

__all__ = [
    "Claimant",
    "RunInbox",
    "TaskId",
    "current_task_id",
]

# A LangGraph task's ID, which a claim is keyed by. The saver gets the same ID as a plain
# string with the task's writes.
TaskId = NewType("TaskId", str)

# Joins a place in the flow's address tree to the ID of one running instance of it, as in
# `developer_agent/review_agent#call-7`.
_INSTANCE_SEPARATOR = "#"


@dataclass(frozen=True, slots=True)
class Claimant:
    """Which pending messages are a task's to claim.

    A task takes the messages and directives addressed to its `recipient`, the unaddressed ones when it
    `accepts_unaddressed`, and the answers whose `in_reply_to` it holds. An address names a place in the flow's tree,
    such as `developer_agent/review_agent`, and one running instance of it adds `#` and an ID. A message addressed to a
    place also reaches its instances, but never the places below it. An answer is found only by its `in_reply_to`,
    never by address. Each selector is optional, so a gate can claim an answer without taking what's addressed to its
    agent.
    """

    recipient: Optional[str] = None
    accepts_unaddressed: bool = False
    in_reply_to: Optional[str] = None

    def takes(self, message: InboundMessage) -> bool:
        """Whether this claimant takes `message`."""
        if isinstance(message, Answer):
            # An answer always has an `in_reply_to`, so a claimant without one takes none.
            return message.in_reply_to == self.in_reply_to
        if message.recipient is None:
            return self.accepts_unaddressed
        if self.recipient is None:
            return False
        return self.recipient == message.recipient or self.recipient.startswith(
            message.recipient + _INSTANCE_SEPARATOR
        )


@dataclass(frozen=True, slots=True)
class _Claim:
    claimant: Claimant
    messages: tuple[InboundMessage, ...]


@dataclass(slots=True)
class _Store:
    loop: asyncio.AbstractEventLoop = field(default_factory=asyncio.get_running_loop)
    # In arrival order.
    pending: list[InboundMessage] = field(default_factory=list)
    # Every ID put, pending or claimed, so a message is delivered once per run.
    seen: set[str] = field(default_factory=set)
    rows: set[str] = field(default_factory=set)
    claims: dict[TaskId, _Claim] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, eq=False)
class RunInbox:
    """The messages waiting for one run, and what each task claimed.

    Made with `RunInbox()` on the run's event loop, once per run, and never stored. LangGraph hands a nested run its
    parent's runtime context, so a nested run claims through the same inbox: it shares the pending messages and
    claims, and what it leaves stays pending for its parent. The inbox doesn't know components: each task says what it
    takes with a `Claimant`, and claims every pending message it takes at once, keyed by its task ID. Claimed messages
    never return to pending, so a sibling never sees them. What nothing claims goes away with the inbox, and a row
    stays pending in the mailbox.

    Every method runs without an `await`, so it's atomic on the run's event loop and needs no lock. A sync node runs in
    a worker thread, where it would race, so the inbox refuses calls from anywhere but the loop that made it.
    """

    _store: _Store = field(default_factory=_Store)

    def put(self, message: InboundMessage, *, row: bool = False) -> None:
        """Deliver a message once per run.

        A message whose ID the run already has isn't delivered again, and the first one put stays. The same ID is the
        same input, so marking it a row still counts.

        Args:
            message: The message to deliver.
            row: Whether it's a mailbox row, which the checkpointer acks with the first durable checkpoint after a task
                that claims it finishes. Anything else, such as a delegation or a resume handed over by a gate, is
                never acked.
        """
        store = self._thread_safe_store()
        if row:
            store.rows.add(message.id)
        if message.id in store.seen:
            return
        store.seen.add(message.id)
        store.pending.append(message)

    def claim(self, task_id: TaskId, claimant: Claimant) -> tuple[InboundMessage, ...]:
        """Take every pending message the claimant takes, answers first and then the rest in arrival order.

        A task claims once. Running again in the same run, it gets the same messages back, an empty claim included,
        even if more arrived since.

        Args:
            task_id: The claiming task's ID, from `current_task_id`.
            claimant: Which messages the task takes.

        Returns:
            The messages the task claimed.

        Raises:
            ValueError: If the task already claimed as a different claimant.
        """
        store = self._thread_safe_store()
        if (claim := store.claims.get(task_id)) is not None:
            if claim.claimant != claimant:
                raise ValueError(
                    f"Task {task_id} claimed as {claim.claimant}, not {claimant}."
                )
            return claim.messages
        taken = [message for message in store.pending if claimant.takes(message)]
        # Stable, so answers come first and each group keeps its arrival order.
        messages = tuple(
            sorted(taken, key=lambda message: not isinstance(message, Answer))
        )
        taken_ids = {message.id for message in messages}
        store.pending = [
            message for message in store.pending if message.id not in taken_ids
        ]
        store.claims[task_id] = _Claim(claimant, messages)
        return messages

    def rows_claimed_by(self, task_id: TaskId) -> tuple[str, ...]:
        """The IDs of the mailbox rows a task claimed, whether or not the task finished.

        A task that fails or pauses acks nothing, so the checkpointer acks them only for a task whose writes hold no
        error or interrupt.
        """
        store = self._thread_safe_store()
        claim = store.claims.get(task_id)
        if claim is None:
            return ()
        return tuple(
            message.id for message in claim.messages if message.id in store.rows
        )

    def _thread_safe_store(self) -> _Store:
        try:
            loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not self._store.loop:
            raise RuntimeError(
                "A run's inbox is used only on the event loop that made it, never from"
                " a sync node or another thread."
            )
        return self._store


def current_task_id(runtime: Runtime[ContextT]) -> TaskId:
    """The ID of the running task, which its claim is keyed by.

    LangGraph derives it from the checkpoint, namespace, step and node, and the task's triggers or `Send` index. So it's
    unique across nested runs and across one node's tasks in a step, and it stays the same when a retry policy runs the
    task again. A nested run under a retried task gets new IDs, though. It's also the ID the saver gets with the task's
    writes.

    Raises:
        RuntimeError: If no graph task is running.
    """
    if runtime.execution_info is None:
        raise RuntimeError("Only a running graph task can claim.")
    return TaskId(runtime.execution_info.task_id)
