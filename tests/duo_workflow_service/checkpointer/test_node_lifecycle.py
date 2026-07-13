from typing import Annotated
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from duo_workflow_service.checkpointer.node_lifecycle import (
    NodeEvent,
    NodeEventLog,
    NodeLifecycleCallbackHandler,
    NodePhase,
)


def _tuples(log: NodeEventLog):
    """Events as ``(component, phase)`` for order-sensitive assertions."""
    return [(e.component, e.phase) for e in log.events]


def _open_run_ids(log: NodeEventLog) -> set[str]:
    """Run ids that started but never reached a terminal event."""
    open_runs: set[str] = set()
    for event in log.events:
        if event.phase == NodePhase.STARTED:
            open_runs.add(event.run_id)
        else:
            open_runs.discard(event.run_id)
    return open_runs


# --- NodeEvent / NodeEventLog (pure state) -----------------------------------


def test_event_to_dict_is_the_wire_shape():
    event = NodeEvent(run_id="run-1", component="researcher", phase=NodePhase.STARTED)

    assert event.to_dict() == {
        "run_id": "run-1",
        "component": "researcher",
        "phase": "started",
    }


def test_log_records_in_order():
    log = NodeEventLog()

    log.record("run-1", "build_context", NodePhase.STARTED)
    log.record("run-1", "build_context", NodePhase.ENDED)
    log.record("run-2", "researcher", NodePhase.STARTED)

    assert _tuples(log) == [
        ("build_context", NodePhase.STARTED),
        ("build_context", NodePhase.ENDED),
        ("researcher", NodePhase.STARTED),
    ]


# --- NodeLifecycleCallbackHandler --------------------------------------------


def _node_start(handler, run_id, node_name, *, boundary=True):
    tags = ["graph:step:1"] if boundary else ["seq:step:1"]
    return handler.on_chain_start(
        {}, {}, run_id=run_id, tags=tags, metadata={"langgraph_node": node_name}
    )


@pytest.mark.asyncio
async def test_handler_records_started_on_node_boundary():
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)
    run_id = uuid4()

    await _node_start(handler, run_id, "researcher#agent")

    assert log.events == [
        NodeEvent(str(run_id), "researcher", NodePhase.STARTED),
    ]


@pytest.mark.asyncio
async def test_handler_records_ended_on_node_end():
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)
    run_id = uuid4()

    await _node_start(handler, run_id, "researcher#agent")
    await handler.on_chain_end({}, run_id=run_id)

    assert _tuples(log) == [
        ("researcher", NodePhase.STARTED),
        ("researcher", NodePhase.ENDED),
    ]
    assert _open_run_ids(log) == set()


@pytest.mark.asyncio
async def test_handler_records_errored_on_non_interrupt_error():
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)
    run_id = uuid4()

    await _node_start(handler, run_id, "researcher#agent")
    await handler.on_chain_error(RuntimeError("boom"), run_id=run_id)

    assert _tuples(log) == [
        ("researcher", NodePhase.STARTED),
        ("researcher", NodePhase.ERRORED),
    ]


@pytest.mark.asyncio
async def test_handler_records_ended_not_errored_on_graph_interrupt():
    # A node pausing for human input raises GraphInterrupt via on_chain_error;
    # it is a pause, not a failure, so it must not be recorded as ERRORED.
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)
    run_id = uuid4()

    await _node_start(handler, run_id, "approval#agent")
    await handler.on_chain_error(GraphInterrupt(), run_id=run_id)

    assert _tuples(log) == [
        ("approval", NodePhase.STARTED),
        ("approval", NodePhase.ENDED),
    ]


@pytest.mark.asyncio
async def test_handler_ignores_nested_runnable_starts():
    # Inner runnables inherit langgraph_node but carry seq:step tags, not
    # graph:step — they must not be recorded as node executions.
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)

    await _node_start(handler, uuid4(), "researcher#agent", boundary=False)

    assert not log.events


@pytest.mark.asyncio
async def test_handler_handles_legacy_node_without_role_suffix():
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)
    run_id = uuid4()

    await _node_start(handler, run_id, "build_context")
    await handler.on_chain_end({}, run_id=run_id)

    assert _tuples(log) == [
        ("build_context", NodePhase.STARTED),
        ("build_context", NodePhase.ENDED),
    ]


@pytest.mark.asyncio
async def test_handler_tracks_concurrent_nodes_by_run_id():
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)
    alpha, beta = uuid4(), uuid4()

    await _node_start(handler, alpha, "alpha#agent")
    await _node_start(handler, beta, "beta#agent")
    await handler.on_chain_end({}, run_id=alpha)

    assert _open_run_ids(log) == {str(beta)}
    assert _tuples(log) == [
        ("alpha", NodePhase.STARTED),
        ("beta", NodePhase.STARTED),
        ("alpha", NodePhase.ENDED),
    ]


@pytest.mark.asyncio
async def test_handler_terminal_for_unopened_run_is_a_noop():
    # Inner runnables and the graph root emit on_chain_end with run ids the
    # handler never opened; those must not produce orphan terminal events.
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)

    await handler.on_chain_end({}, run_id=uuid4())
    await handler.on_chain_error(RuntimeError("boom"), run_id=uuid4())

    assert not log.events


@pytest.mark.asyncio
async def test_handler_ignores_start_with_no_component():
    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)

    await handler.on_chain_start(
        {}, {}, run_id=uuid4(), tags=["graph:step:1"], metadata={"langgraph_node": ""}
    )
    await handler.on_chain_start(
        {}, {}, run_id=uuid4(), tags=["graph:step:1"], metadata={}
    )

    assert not log.events


# --- Regression guards against real LangGraph --------------------------------


@pytest.mark.asyncio
async def test_handler_against_real_langgraph_stream():
    # Guards the LangGraph contract the boundary discriminator relies on: node
    # boundaries carry the graph:step tag while inner runnables do not. Mirrors
    # v1 node naming, includes a silent node (no '#', no output message), and
    # runs the handler through a real astream.
    class State(TypedDict):
        steps: Annotated[list, lambda a, b: a + b]

    def build_context(_state: State) -> dict:
        return {"steps": ["build_context"]}

    def researcher_agent(_state: State) -> dict:
        # nested runnable inside the node — must NOT be recorded as a node
        RunnableLambda(lambda x: x, name="inner").invoke({})
        return {"steps": ["researcher#agent"]}

    graph = StateGraph(State)
    graph.add_node("build_context", build_context)
    graph.add_node("researcher#agent", researcher_agent)
    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", "researcher#agent")
    graph.add_edge("researcher#agent", END)
    compiled = graph.compile()

    log = NodeEventLog()
    handler = NodeLifecycleCallbackHandler(log)

    async for _ in compiled.astream(
        {"steps": []}, stream_mode=["values"], config={"callbacks": [handler]}
    ):
        pass

    # Both nodes (including the silent one) recorded and mapped to components,
    # each cleanly started then ended; the nested "inner" runnable did not leak;
    # nothing left open.
    assert _tuples(log) == [
        ("build_context", NodePhase.STARTED),
        ("build_context", NodePhase.ENDED),
        ("researcher", NodePhase.STARTED),
        ("researcher", NodePhase.ENDED),
    ]
    assert _open_run_ids(log) == set()


@pytest.mark.asyncio
async def test_handler_against_real_langgraph_interrupt_and_resume():
    # Guards the human-in-the-loop contract: a node that calls interrupt() pauses
    # via GraphInterrupt (recorded ENDED, never ERRORED, never left open), and on
    # resume the node re-runs under a *new* run id. The notifier creates a fresh
    # log per run segment, so we mirror that with a second log on resume.
    class State(TypedDict):
        steps: Annotated[list, lambda a, b: a + b]

    def build_context(_state: State) -> dict:
        return {"steps": ["build_context"]}

    def approval_agent(_state: State) -> dict:
        decision = interrupt({"ask": "approve?"})
        return {"steps": [f"approval:{decision}"]}

    def finalize(_state: State) -> dict:
        return {"steps": ["finalize"]}

    graph = StateGraph(State)
    graph.add_node("build_context", build_context)
    graph.add_node("approval#agent", approval_agent)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "build_context")
    graph.add_edge("build_context", "approval#agent")
    graph.add_edge("approval#agent", "finalize")
    graph.add_edge("finalize", END)
    compiled = graph.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    # Segment 1: run until the interrupt.
    log1 = NodeEventLog()
    async for _ in compiled.astream(
        {"steps": []},
        stream_mode=["values"],
        config={**config, "callbacks": [NodeLifecycleCallbackHandler(log1)]},
    ):
        pass

    assert _tuples(log1) == [
        ("build_context", NodePhase.STARTED),
        ("build_context", NodePhase.ENDED),
        ("approval", NodePhase.STARTED),
        ("approval", NodePhase.ENDED),  # interrupt -> ENDED, not ERRORED
    ]
    assert all(e.phase != NodePhase.ERRORED for e in log1.events)
    assert _open_run_ids(log1) == set()
    approval_run_seg1 = next(e.run_id for e in log1.events if e.component == "approval")

    # Segment 2: resume with a fresh log (as a resumed run does in production).
    log2 = NodeEventLog()
    async for _ in compiled.astream(
        Command(resume="yes"),
        stream_mode=["values"],
        config={**config, "callbacks": [NodeLifecycleCallbackHandler(log2)]},
    ):
        pass

    assert _tuples(log2) == [
        ("approval", NodePhase.STARTED),
        ("approval", NodePhase.ENDED),
        ("finalize", NodePhase.STARTED),
        ("finalize", NodePhase.ENDED),
    ]
    assert _open_run_ids(log2) == set()
    approval_run_seg2 = next(e.run_id for e in log2.events if e.component == "approval")
    # The resumed node is a distinct run, so clients never collide the two.
    assert approval_run_seg1 != approval_run_seg2
