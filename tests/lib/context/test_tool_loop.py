"""Tests for lib/context/tool_loop module."""

import asyncio
import contextvars

import pytest

from lib.context.tool_loop import (
    ToolLoopStats,
    build_tool_loop_session_summary_extras,
    init_tool_loop_counters,
    record_tool_calls,
    tool_loop_stats,
)

GREP = ("grep", {"q": "x"}, "hit")
READ = ("read_file", {"path": "a.py"}, "body")


@pytest.fixture(autouse=True)
def reset_counters():
    tool_loop_stats.set(None)
    yield
    tool_loop_stats.set(None)


@pytest.mark.parametrize(
    ("calls", "expected"),
    [
        pytest.param([], (0, 0, 0, 0), id="nothing-recorded"),
        pytest.param([GREP], (1, 1, 1, 1), id="single-call-is-a-streak-of-one"),
        pytest.param(
            [("read_file", {"path": p}, "body") for p in ("a.py", "b.py", "c.py")],
            (3, 3, 1, 1),
            id="distinct-calls-build-no-streak",
        ),
        pytest.param(
            [("run_command", {"cmd": "git fetch"}, "fatal")] * 4,
            (4, 1, 4, 4),
            id="identical-call-and-result-builds-both-streaks",
        ),
        pytest.param(
            [
                ("get_pipeline", {"id": 1}, s)
                for s in ("pending", "running", "running", "success")
            ],
            (4, 1, 4, 2),
            id="changing-result-breaks-only-the-result-streak",
        ),
        pytest.param([GREP, GREP, READ, GREP], (4, 2, 2, 2), id="streak-resets"),
        pytest.param(
            [GREP, GREP, GREP, READ],
            (4, 2, 3, 3),
            id="max-retained-after-streak-ends",
        ),
        pytest.param(
            [("grep", {"q": "x"}, "hit"), ("grep", {"q": "y"}, "hit")],
            (2, 2, 1, 1),
            id="same-tool-different-args-is-not-a-repeat",
        ),
        pytest.param(
            [
                ("read_file", {"path": "a.py"}, "body"),
                ("list_dir", {"path": "a.py"}, "body"),
            ],
            (2, 2, 1, 1),
            id="same-args-different-tool-is-not-a-repeat",
        ),
        pytest.param(
            [
                ("edit_file", {"a": 1, "b": 2}, "ok"),
                ("edit_file", {"b": 2, "a": 1}, "ok"),
            ],
            (2, 1, 2, 2),
            id="argument-key-order-does-not-affect-the-hash",
        ),
        pytest.param(
            [("get_issue", {"id": 1}, [{"type": "text", "text": "body"}])] * 2,
            (2, 1, 2, 2),
            id="structured-results-are-comparable",
        ),
        pytest.param(
            # values are covered by json.dumps(default=str)
            [("weird_tool", {"when": object()}, "ok")],
            (1, 1, 1, 1),
            id="non-serialisable-arg-value-does-not-raise",
        ),
        pytest.param(
            # keys are not: default= never applies to them, so sort_keys trips
            # over the mixed types and the hash falls back to str()
            [("get_issue", {"id": 1}, [{"type": "text", 1: "body"}])],
            (1, 1, 1, 1),
            id="mixed-type-result-keys-do-not-raise",
        ),
        pytest.param(
            [("get_issue", {"id": 1}, {"outer": {"type": "text", None: "body"}})],
            (1, 1, 1, 1),
            id="nested-mixed-type-result-keys-do-not-raise",
        ),
        pytest.param(
            [("weird_tool", {1: "a", "b": 2}, "ok")],
            (1, 1, 1, 1),
            id="mixed-type-arg-keys-do-not-raise",
        ),
        pytest.param(
            # the fallback must still recognise a repeat, not just avoid raising
            [("get_issue", {"id": 1}, [{"type": "text", 1: "body"}])] * 3,
            (3, 1, 3, 3),
            id="fallback-still-counts-repeats",
        ),
        pytest.param(
            [("run_command", {"cmd": "git fetch"}, "fatal")] * 230,
            (230, 1, 230, 230),
            id="runaway-loop-teleport-78b0",
        ),
        pytest.param(
            [("read_file", {"path": f"f{i}.py"}, f"body{i}") for i in range(290)],
            (290, 290, 1, 1),
            id="productive-sweep-must-not-look-like-a-loop",
        ),
        pytest.param(
            [("a", {}, "1"), ("b", {}, "2")] * 50,
            (100, 2, 1, 1),
            id="alternating-loop-is-invisible-to-consecutive-metrics",
        ),
    ],
)
def test_stats_arithmetic(calls, expected):
    """Fold a call sequence and check (total, unique, max_args, max_args_result)."""
    stats = ToolLoopStats()

    for call in calls:
        stats.record(*call)

    assert (
        stats.total_calls,
        stats.unique_calls,
        stats.max_consecutive_args,
        stats.max_consecutive_args_result,
    ) == expected


def test_unserialisable_payloads_are_matched_by_key_order():
    """Documents a known imprecision rather than asserting desirable behaviour.

    The hash falls back to ``str()`` for payloads JSON cannot sort, and that
    rendering is order-sensitive, so two dicts with identical content but
    different insertion order look like different calls. This can only
    under-count a loop, never invent one, which is the safe direction.
    """
    stats = ToolLoopStats()

    stats.record("get_issue", {}, {"a": 1, 1: 2})
    stats.record("get_issue", {}, {1: 2, "a": 1})

    assert stats.max_consecutive_args_result == 1


def test_init_installs_fresh_stats():
    init_tool_loop_counters()

    assert tool_loop_stats.get().total_calls == 0


def test_init_discards_counters_from_a_previous_session():
    init_tool_loop_counters()
    record_tool_calls([GREP])

    init_tool_loop_counters()

    assert tool_loop_stats.get().total_calls == 0


def test_record_is_a_no_op_when_never_initialised():
    record_tool_calls([GREP])  # must not raise

    assert tool_loop_stats.get() is None


@pytest.mark.parametrize(
    "batches",
    [
        pytest.param([[GREP, GREP]], id="within-one-batch"),
        pytest.param([[GREP], [GREP]], id="across-batches"),
        pytest.param([[GREP], [], [GREP]], id="empty-batch-does-not-break-the-streak"),
    ],
)
def test_streaks_span_batches(batches):
    """Each ToolNode.run is one batch, so a real loop spans many of them."""
    init_tool_loop_counters()

    for batch in batches:
        record_tool_calls(batch)

    assert tool_loop_stats.get().max_consecutive_args == 2


async def _gather(call):
    await asyncio.gather(call(), call(), call())


async def _copied_context(call):
    for _ in range(3):
        ctx = contextvars.copy_context()
        await ctx.run(asyncio.create_task, call())
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatch", [_gather, _copied_context], ids=["gather", "copy_context"]
)
async def test_updates_from_copied_contexts_reach_the_parent(dispatch):
    """Nested updates must reach the parent's counters.

    ``ToolNode`` gathers its calls and supervisor_v2 dispatches subagents via
    ``Send``; both hand the callee a copied context. Stats are a mutable object
    precisely so those updates are not lost.
    """
    init_tool_loop_counters()

    async def call():
        record_tool_calls([GREP])

    await dispatch(call)

    assert tool_loop_stats.get().total_calls == 3
    assert tool_loop_stats.get().max_consecutive_args == 3


@pytest.mark.parametrize(
    "initialise",
    [
        pytest.param(False, id="never-initialised"),
        pytest.param(True, id="no-tools-ran"),
    ],
)
def test_summary_is_none_when_there_is_nothing_to_report(initialise):
    if initialise:
        init_tool_loop_counters()

    assert (
        build_tool_loop_session_summary_extras(
            workflow_id="wf-1", workflow_type="developer"
        )
        is None
    )


def test_summary_reports_every_property():
    init_tool_loop_counters()
    record_tool_calls([GREP, GREP, READ])

    result = build_tool_loop_session_summary_extras(
        workflow_id="wf-42", workflow_type="developer"
    )

    assert result == {
        "value": "wf-42",
        "workflow_type": "developer",
        "total_tool_calls": 3,
        "unique_tool_calls": 2,
        "max_consecutive_identical_calls": 2,
        "max_consecutive_identical_with_result": 2,
    }
