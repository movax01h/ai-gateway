"""Tests for the identical tool-call detection helpers.

Behaviour reachable through ``ToolNode`` is covered by
``TestToolNodeIdenticalCallGuard`` in ``test_v1_tool_node.py``, driven over
successive node executions as the guard is actually used. Only the guard clauses
the node cannot reach — it always inspects a history ending in an ``AIMessage``
with tool calls — are asserted directly here.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.components.agent.nodes._loop_detection import (
    IdenticalCallChain,
    inspect_identical_call_chain,
)


class TestInspectIdenticalCallChainWithoutTrailingToolCalls:
    """No trailing tool-call batch means there is no chain to measure."""

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param([], id="empty_history"),
            pytest.param([HumanMessage(content="do the thing")], id="human_message"),
            pytest.param([AIMessage(content="all done")], id="text_only_ai_message"),
            pytest.param(
                [ToolMessage(content="result", tool_call_id="call_0")],
                id="tool_message",
            ),
        ],
    )
    def test_returns_empty_chain(self, history):
        assert inspect_identical_call_chain(history) == IdenticalCallChain(
            repeats=0, results_identical=False, nudges=0
        )


def _call(cycle: int) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "read_file", "args": {"path": "x"}, "id": f"call_{cycle}"}
        ],
    )


class TestInspectIdenticalCallChainResultsIdentical:
    """``results_identical`` needs two executed results; repetition alone must never count as identical.

    A ``ToolNode`` with a valid limit (>= 3) always has two results by the time ``repeats`` reaches it, so the
    fewer-than-two cases are only reachable by callers bypassing the component validator.
    """

    @pytest.mark.parametrize(
        ("history", "expected"),
        [
            pytest.param(
                [_call(0)],
                IdenticalCallChain(repeats=1, results_identical=False, nudges=0),
                id="first_call_no_results",
            ),
            pytest.param(
                [_call(0), ToolMessage(content="err", tool_call_id="call_0"), _call(1)],
                IdenticalCallChain(repeats=2, results_identical=False, nudges=0),
                id="one_result_nothing_to_compare",
            ),
            pytest.param(
                [
                    _call(0),
                    ToolMessage(content="same error", tool_call_id="call_0"),
                    _call(1),
                    ToolMessage(content="same error", tool_call_id="call_1"),
                    _call(2),
                ],
                IdenticalCallChain(repeats=3, results_identical=True, nudges=0),
                id="two_identical_results",
            ),
            pytest.param(
                [
                    _call(0),
                    ToolMessage(content="pending", tool_call_id="call_0"),
                    _call(1),
                    ToolMessage(content="running", tool_call_id="call_1"),
                    _call(2),
                ],
                IdenticalCallChain(repeats=3, results_identical=False, nudges=0),
                id="two_different_results",
            ),
        ],
    )
    def test_results_identical(self, history, expected):
        assert inspect_identical_call_chain(history) == expected
