import asyncio
from typing import Any, Optional, Sequence, TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, Send

from duo_workflow_service.agent_platform.core.inbox import (
    Answer,
    Claimant,
    Component,
    Decision,
    Directive,
    InboundMessage,
    Message,
    RunInbox,
    TaskId,
    TextPart,
    User,
    current_task_id,
)

DEVELOPER = Claimant(recipient="developer", accepts_unaddressed=True)


def _task(id: str) -> TaskId:
    return TaskId(id)


TASK = _task("task-1")


def _message(
    id: str, recipient: Optional[str] = None, *, sender: User | Component = User()
) -> Message:
    return Message(
        id=id,
        content=(TextPart(type="text", text=f"Text of {id}"),),
        sender=sender,
        recipient=recipient,
    )


def _answer(id: str, in_reply_to: str) -> Answer:
    return Answer(
        id=id,
        in_reply_to=in_reply_to,
        content=(TextPart(type="text", text="Not that file"),),
        decision=Decision(choice="reject"),
    )


def _directive(id: str, recipient: Optional[str] = None) -> Directive:
    return Directive(id=id, text="/compact", recipient=recipient)


def _inbox(*messages: InboundMessage, rows: Sequence[str] = ()) -> RunInbox:
    inbox = RunInbox()
    for message in messages:
        inbox.put(message, row=message.id in rows)
    return inbox


def _ids(messages: Sequence[InboundMessage]) -> tuple[str, ...]:
    return tuple(message.id for message in messages)


class TestClaim:
    PENDING = (
        _message("to-developer", "developer"),
        _message("unaddressed"),
        _answer("reply-to-ai-1", "ai-1"),
        _directive("compact", "developer"),
        _message("to-reviewer", "reviewer"),
        _answer("reply-to-ai-2", "ai-2"),
        _answer("second-reply-to-ai-1", "ai-1"),
    )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("claimant", "expected"),
        [
            (Claimant(recipient="developer"), ("to-developer", "compact")),
            (DEVELOPER, ("to-developer", "unaddressed", "compact")),
            (Claimant(accepts_unaddressed=True), ("unaddressed",)),
            (
                Claimant(recipient="developer", in_reply_to="ai-1"),
                ("reply-to-ai-1", "second-reply-to-ai-1", "to-developer", "compact"),
            ),
            (Claimant(in_reply_to="ai-1"), ("reply-to-ai-1", "second-reply-to-ai-1")),
            (Claimant(), ()),
        ],
        ids=[
            "messages and directives addressed to it",
            "unaddressed ones too, in arrival order",
            "an answer never counts as unaddressed",
            "its answers first, then the rest",
            "only answers, leaving its agent's messages",
            "nothing",
        ],
    )
    async def test_a_claimant_takes_what_it_selects(
        self, claimant: Claimant, expected: tuple[str, ...]
    ):
        inbox = _inbox(*self.PENDING)

        assert _ids(inbox.claim(TASK, claimant)) == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("recipient", "expected"),
        [
            ("developer_agent", ("to-developer-agent",)),
            ("developer_agent/review_agent", ("to-review-agent",)),
            ("developer_agent/review_agent#call-7", ("to-review-agent", "to-call-7")),
            ("developer_agent/review", ("to-review",)),
        ],
        ids=[
            "a place, not the places below it",
            "a place, not what's addressed to one of its instances",
            "an instance, and what's addressed to its place",
            "a place whose name another place's starts with",
        ],
    )
    async def test_an_address_reaches_its_place_and_its_instances(
        self, recipient: str, expected: tuple[str, ...]
    ):
        inbox = _inbox(
            _message("to-developer-agent", "developer_agent"),
            _message("to-review", "developer_agent/review"),
            _message("to-review-agent", "developer_agent/review_agent"),
            _message("to-call-7", "developer_agent/review_agent#call-7"),
            _message("to-call-8", "developer_agent/review_agent#call-8"),
        )

        assert _ids(inbox.claim(TASK, Claimant(recipient=recipient))) == expected

    @pytest.mark.asyncio
    async def test_a_claimed_message_is_gone_for_every_other_task(self):
        inbox = _inbox(*self.PENDING)
        inbox.claim(_task("agent"), DEVELOPER)

        assert inbox.claim(_task("sibling"), DEVELOPER) == ()
        assert _ids(inbox.claim(_task("reviewer"), Claimant(recipient="reviewer"))) == (
            "to-reviewer",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("pending", "expected"),
        [((_message("m-1"),), ("m-1",)), ((), ())],
        ids=["a claim", "an empty claim"],
    )
    async def test_a_task_claiming_again_gets_the_same_messages(
        self, pending: tuple[InboundMessage, ...], expected: tuple[str, ...]
    ):
        inbox = _inbox(*pending)
        claimed = inbox.claim(TASK, DEVELOPER)
        inbox.put(_message("steer-1"))

        # A rerun builds its task and claimant again.
        again = Claimant(recipient="developer", accepts_unaddressed=True)
        assert _ids(claimed) == expected
        assert _ids(inbox.claim(_task("task-1"), again)) == expected
        assert _ids(inbox.claim(_task("next-step"), DEVELOPER)) == ("steer-1",)

    @pytest.mark.asyncio
    async def test_a_task_claims_as_one_claimant(self):
        inbox = _inbox()
        inbox.claim(TASK, DEVELOPER)

        with pytest.raises(ValueError, match="claimed as"):
            inbox.claim(TASK, Claimant(recipient="reviewer"))


class TestPut:
    @pytest.mark.asyncio
    async def test_a_pending_id_stays_as_it_was_first_put(self):
        inbox = _inbox(_message("m-1"))

        inbox.put(_message("m-1", "developer"))

        # The first put had no address, so a claimant that only takes its own address finds nothing.
        assert _ids(inbox.claim(TASK, Claimant(recipient="developer"))) == ()
        assert _ids(inbox.claim(_task("task-2"), DEVELOPER)) == ("m-1",)

    @pytest.mark.asyncio
    async def test_a_claimed_id_is_not_delivered_again(self):
        inbox = _inbox(_message("m-1"))
        inbox.claim(TASK, DEVELOPER)

        inbox.put(_message("m-1", "developer"))

        assert _ids(inbox.claim(_task("task-2"), DEVELOPER)) == ()


class TestRowsClaimedBy:
    @pytest.mark.asyncio
    async def test_it_filters_only_mailbox_rows(self):
        delegation = _message(
            "call-1", "developer", sender=Component(name="supervisor")
        )
        inbox = _inbox(
            _message("row-1"),
            delegation,
            _answer("handover-1", "ai-1"),
            _message("row-2", "developer"),
            rows=("row-1", "row-2"),
        )
        task_claims = inbox.claim(
            TASK, Claimant(recipient="developer", in_reply_to="ai-1")
        )
        inbox.claim(_task("task-2"), Claimant(accepts_unaddressed=True))

        assert inbox.rows_claimed_by(TASK) == ("row-2",)
        assert _ids(task_claims) == (
            "handover-1",
            "call-1",
            "row-2",
        )
        assert inbox.rows_claimed_by(TaskId("task-2")) == ("row-1",)
        assert inbox.rows_claimed_by(TaskId("unknown")) == ()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rows", [(True, False), (False, True)], ids=["first", "later"]
    )
    async def test_an_inbound_message_is_returned_whichever_put_marks_it_as_a_row(
        self, rows: tuple[bool, bool]
    ):
        inbox = _inbox()
        for row in rows:
            inbox.put(_message("row-1"), row=row)
        inbox.claim(TASK, DEVELOPER)

        assert inbox.rows_claimed_by(TASK) == ("row-1",)


class TestEventLoop:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "on_another_loop", [False, True], ids=["a worker thread", "another event loop"]
    )
    async def test_only_the_loop_that_made_the_inbox_can_use_it(
        self, on_another_loop: bool
    ):
        inbox = _inbox()

        def claim() -> tuple[InboundMessage, ...]:
            return inbox.claim(TASK, DEVELOPER)

        async def claim_on_its_loop() -> tuple[InboundMessage, ...]:
            return claim()

        with pytest.raises(RuntimeError, match="event loop that made it"):
            if on_another_loop:
                await asyncio.to_thread(asyncio.run, claim_on_its_loop())
            else:
                await asyncio.to_thread(claim)


class _State(TypedDict, total=False):
    done: bool


class _RecordingSaver(InMemorySaver):
    def __init__(self) -> None:
        super().__init__()
        self.task_ids: list[str] = []

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.task_ids.append(task_id)
        await super().aput_writes(config, writes, task_id, task_path)


class TestClaimsInAGraph:
    """Guards for what claims rely on in the pinned LangGraph: what a task ID is, and what a nested run inherits."""

    @pytest.mark.asyncio
    async def test_a_retried_task_reads_its_claim_back(self):
        inbox = _inbox(_message("m-1"))
        claims: list[tuple[InboundMessage, ...]] = []

        async def agent(state: _State, *, runtime: Runtime) -> _State:
            claimant = Claimant(recipient="developer", accepts_unaddressed=True)
            claims.append(inbox.claim(current_task_id(runtime), claimant))
            if len(claims) == 1:
                inbox.put(_message("steer-1"))
                raise ConnectionError("The model call failed.")
            return {"done": True}

        graph = StateGraph(_State)
        graph.add_node(
            "agent",
            agent,
            retry_policy=RetryPolicy(
                initial_interval=0, jitter=False, retry_on=ConnectionError
            ),
        )
        graph.add_edge(START, "agent")
        await graph.compile().ainvoke({})

        assert [_ids(claim) for claim in claims] == [("m-1",), ("m-1",)]
        assert _ids(inbox.claim(_task("next-step"), DEVELOPER)) == ("steer-1",)

    @pytest.mark.asyncio
    async def test_tasks_of_one_node_in_one_step_claim_apart(self):
        inbox = _inbox(_message("m-1"), _message("m-2"))
        claims: list[tuple[InboundMessage, ...]] = []

        async def agent(state: _State, *, runtime: Runtime) -> _State:
            claims.append(inbox.claim(current_task_id(runtime), DEVELOPER))
            return {}

        graph = StateGraph(_State)
        graph.add_node("agent", agent)
        graph.add_conditional_edges(
            START, lambda _state: [Send("agent", {}), Send("agent", {})]
        )
        await graph.compile().ainvoke({})

        assert sorted(_ids(claim) for claim in claims) == [(), ("m-1", "m-2")]

    @pytest.mark.asyncio
    async def test_a_claim_is_keyed_by_the_task_id_the_saver_gets(self):
        inbox = _inbox(
            _message("to-supervisor", "supervisor"),
            _message("to-developer", "developer"),
            rows=("to-supervisor", "to-developer"),
        )
        tasks: dict[str, TaskId] = {}

        async def subagent(state: _State, *, runtime: Runtime) -> _State:
            tasks["subagent"] = current_task_id(runtime)
            inbox.claim(tasks["subagent"], DEVELOPER)
            return {"done": True}

        nested = StateGraph(_State)
        nested.add_node("subagent", subagent)
        nested.add_edge(START, "subagent")
        nested_graph = nested.compile()

        async def dispatch(state: _State, *, runtime: Runtime) -> _State:
            tasks["dispatch"] = current_task_id(runtime)
            inbox.claim(tasks["dispatch"], Claimant(recipient="supervisor"))
            await nested_graph.ainvoke({})
            return {"done": True}

        graph = StateGraph(_State)
        graph.add_node("dispatch", dispatch)
        graph.add_edge(START, "dispatch")
        saver = _RecordingSaver()
        await graph.compile(checkpointer=saver).ainvoke(
            {}, {"configurable": {"thread_id": "session-1"}}
        )

        assert tasks["dispatch"] != tasks["subagent"]
        assert set(tasks.values()) <= set(saver.task_ids)
        assert inbox.rows_claimed_by(tasks["dispatch"]) == ("to-supervisor",)
        assert inbox.rows_claimed_by(tasks["subagent"]) == ("to-developer",)

    @pytest.mark.asyncio
    async def test_a_nested_run_claims_through_the_inbox_it_inherits(self):
        inbox = _inbox(
            _message("to-developer", "developer"),
            _message("unaddressed"),
            rows=("to-developer",),
        )
        nested_inboxes: list[RunInbox] = []
        nested_tasks: list[TaskId] = []

        async def subagent(state: _State, *, runtime: Runtime[RunInbox]) -> _State:
            nested_inboxes.append(runtime.context)
            nested_tasks.append(current_task_id(runtime))
            runtime.context.claim(nested_tasks[0], Claimant(recipient="developer"))
            return {"done": True}

        nested = StateGraph(_State, context_schema=RunInbox)
        nested.add_node("subagent", subagent)
        nested.add_edge(START, "subagent")
        nested_graph = nested.compile()

        async def dispatch(state: _State) -> _State:
            # A nested site that passes no `context=` inherits the parent's.
            await nested_graph.ainvoke({})
            return {"done": True}

        graph = StateGraph(_State, context_schema=RunInbox)
        graph.add_node("dispatch", dispatch)
        graph.add_edge(START, "dispatch")
        await graph.compile().ainvoke({}, context=inbox)

        assert nested_inboxes == [inbox]
        assert inbox.rows_claimed_by(nested_tasks[0]) == ("to-developer",)
        assert _ids(inbox.claim(_task("entry"), DEVELOPER)) == ("unaddressed",)

    def test_only_a_running_task_has_an_id(self):
        with pytest.raises(RuntimeError, match="Only a running graph task can claim."):
            current_task_id(Runtime())
