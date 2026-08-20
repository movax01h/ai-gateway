"""Test suite for SubagentDispatchNode."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt, GraphRecursionError, ParentCommand
from langgraph.types import Command

from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    SUBSESSION_ID_CONTEXT_KEY,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node import (
    AgentStuckError,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DELEGATION_CALL_ID_CONTEXT_KEY,
    DelegationFatalError,
    DelegationStatus,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.subagent_dispatch_node import (
    SubagentDispatchNode,
)
from duo_workflow_service.agent_platform.v1.state import IOKey
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorType
from duo_workflow_service.errors.typing import (
    InvalidRequestException,
    NotifiableException,
)
from duo_workflow_service.security.exceptions import SecurityException
from lib.usage_quota.errors import InsufficientEntitlements
from lib.usage_quota.service import InsufficientCredits

_CALL_ID = "delegate_call_123"


@pytest.fixture(name="compiled_graph")
def compiled_graph_fixture():
    """Fixture for a mock compiled subagent subgraph."""
    graph = Mock()
    graph.ainvoke = AsyncMock()
    return graph


@pytest.fixture(name="dispatch_node")
def dispatch_node_fixture(
    compiled_graph,
    developer_name,
    subsession_run_key_factory,
):
    """Fixture for a SubagentDispatchNode wired to the mock compiled graph."""
    return SubagentDispatchNode(
        subagent_name=developer_name,
        subsession_run_key_factory=subsession_run_key_factory,
        compiled_graph=compiled_graph,
    )


def _dispatched_state(subsession_id=1, call_id=_CALL_ID) -> dict:
    """Build an initial state as ``DelegationPrepareNode`` seeds it for one dispatch."""
    context = {}
    if subsession_id is not None:
        context[SUBSESSION_ID_CONTEXT_KEY] = subsession_id
    if call_id is not None:
        context[DELEGATION_CALL_ID_CONTEXT_KEY] = call_id
    return {
        "status": None,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": context,
        "agent_context_limits": {},
    }


def _subagent_result(
    subagent_name: str,
    *,
    history: list | None = None,
    context: dict | None = None,
    ui_chat_log: list | None = None,
) -> dict:
    """Build what a dispatched subagent's own ``ainvoke`` returns.

    Every channel is present because ``DelegationPrepareNode._build_initial_state``
    seeds all of them on the dispatched initial state, so LangGraph has a value
    for each one to return.
    """
    return {
        "status": None,
        "conversation_history": {subagent_name: history if history is not None else []},
        "ui_chat_log": ui_chat_log if ui_chat_log is not None else [],
        "context": {subagent_name: context if context is not None else {}},
        "agent_context_limits": {},
    }


def _run_record(result: dict, supervisor_name: str, call_id: str = _CALL_ID) -> dict:
    """Read back the run record ``result`` recorded for ``call_id``."""
    return result["context"][supervisor_name]["subsession_runs"][call_id]


class TestSubagentDispatchNodeSuccess:
    """Tests for the success path of SubagentDispatchNode.run."""

    @pytest.mark.asyncio
    async def test_records_the_run_and_the_subagents_ui_entries(
        self, dispatch_node, compiled_graph, developer_name, supervisor_name
    ):
        compiled_graph.ainvoke.return_value = _subagent_result(
            developer_name,
            history=[AIMessage(content="Done.")],
            context={"final_answer": "Done."},
            ui_chat_log=[{"content": "working..."}],
        )

        result = await dispatch_node.run(_dispatched_state(1), config={})

        assert _run_record(result, supervisor_name) == {
            "subsession_id": 1,
            "status": DelegationStatus.COMPLETED,
            "error": None,
            "final_answer": "Done.",
        }
        assert result["ui_chat_log"] == [{"content": "working..."}]

    @pytest.mark.asyncio
    async def test_never_copies_the_subagents_transcript_into_the_parent(
        self, dispatch_node, compiled_graph, developer_name
    ):
        """Nothing reads it back once a subsession cannot be resumed, so carrying it would grow the parent's checkpoint
        by a transcript per delegation for no reader."""
        compiled_graph.ainvoke.return_value = _subagent_result(
            developer_name,
            history=[AIMessage(content="a long subagent transcript")],
            context={"final_answer": "Done."},
        )

        result = await dispatch_node.run(_dispatched_state(1), config={})

        assert "conversation_history" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raises,expected_status,expected_error,expected_answer",
        [
            pytest.param(
                False, DelegationStatus.COMPLETED, None, "Done.", id="success"
            ),
            pytest.param(True, DelegationStatus.ERROR, "boom", None, id="failure"),
        ],
    )
    async def test_writes_land_wherever_the_factories_point(
        self,
        compiled_graph,
        developer_name,
        raises,
        expected_status,
        expected_error,
        expected_answer,
    ):
        """The node writes through its factories' keys, not conventions of its own.

        The factory is the supervisor's single definition of where a run record
        lives, so a node that reproduced that convention locally could drift
        from what ``DelegationCollectNode`` reads back. A deliberately
        unconventional factory proves neither the success nor the error path
        spells the key itself.
        """
        node = SubagentDispatchNode(
            subagent_name=developer_name,
            subsession_run_key_factory=lambda call_id: IOKey(
                target="context", subkeys=["runs", call_id]
            ),
            compiled_graph=compiled_graph,
        )
        if raises:
            compiled_graph.ainvoke.side_effect = RuntimeError("boom")
        else:
            compiled_graph.ainvoke.return_value = _subagent_result(
                developer_name, context={"final_answer": "Done."}
            )

        result = await node.run(_dispatched_state(7), config={})

        assert result["context"]["runs"][_CALL_ID] == {
            "subsession_id": 7,
            "status": expected_status,
            "error": expected_error,
            "final_answer": expected_answer,
        }

    @pytest.mark.asyncio
    async def test_missing_final_answer_is_recorded_as_a_completed_run_without_one(
        self, dispatch_node, compiled_graph, developer_name, supervisor_name
    ):
        """Reaching the subagent's terminal node is COMPLETED even with nothing to show.

        Reporting an answerless run back to the supervisor is
        ``DelegationCollectNode``'s job, so this node records the plain facts:
        the run finished, and produced no answer.
        """
        compiled_graph.ainvoke.return_value = _subagent_result(developer_name)

        result = await dispatch_node.run(_dispatched_state(1), config={})

        record = _run_record(result, supervisor_name)
        assert record["status"] == DelegationStatus.COMPLETED
        assert record["final_answer"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "stale,fresh_result_context,expected_error,expected_answer",
        [
            pytest.param(
                {
                    "subsession_id": 1,
                    "status": DelegationStatus.ERROR,
                    "error": "boom",
                    "final_answer": None,
                },
                {"final_answer": "Done on retry."},
                None,
                "Done on retry.",
                id="successful-rerun-clears-the-previous-runs-error",
            ),
            pytest.param(
                {
                    "subsession_id": 1,
                    "status": DelegationStatus.COMPLETED,
                    "error": None,
                    "final_answer": "Answer from the first run.",
                },
                {},
                None,
                None,
                id="answerless-rerun-clears-the-previous-runs-answer",
            ),
        ],
    )
    async def test_run_fully_replaces_a_replayed_calls_earlier_record(
        self,
        dispatch_node,
        compiled_graph,
        developer_name,
        supervisor_name,
        stale,
        fresh_result_context,
        expected_error,
        expected_answer,
    ):
        """Every field is written on every dispatch, so a re-run replaces the record whole.

        ``context`` deep-merges, so a field left out of a re-run of the same
        call (a replayed ``Send`` after an interrupt) would keep the previous
        run's value: a stale error would mask a successful retry, and a stale
        final_answer would report a retry that produced no answer as having
        completed with the earlier answer.
        """
        compiled_graph.ainvoke.return_value = _subagent_result(
            developer_name, context=fresh_result_context
        )

        state = _dispatched_state(1)
        # An earlier run of this same call already merged its record in.
        state["context"][supervisor_name] = {"subsession_runs": {_CALL_ID: dict(stale)}}

        result = await dispatch_node.run(state, config={})

        record = _run_record(result, supervisor_name)
        assert record == {
            "subsession_id": 1,
            "status": DelegationStatus.COMPLETED,
            "error": expected_error,
            "final_answer": expected_answer,
        }
        # The update must overwrite every field the stale record had set.
        assert set(stale) == set(record)

    @pytest.mark.asyncio
    async def test_passes_state_and_config_through_to_ainvoke(
        self, dispatch_node, compiled_graph, developer_name
    ):
        compiled_graph.ainvoke.return_value = _subagent_result(developer_name)
        state = _dispatched_state(1)
        config = {"configurable": {"thread_id": "abc"}}

        await dispatch_node.run(state, config=config)

        compiled_graph.ainvoke.assert_awaited_once_with(state, config=config)


class TestSubagentDispatchNodeErrorHandling:
    """Tests for SubagentDispatchNode.run's error handling."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state",
        [
            pytest.param(_dispatched_state(subsession_id=None), id="no-subsession-id"),
            pytest.param(_dispatched_state(call_id=None), id="no-call-id"),
        ],
    )
    async def test_missing_seeded_value_raises_fatal_error(
        self, dispatch_node, compiled_graph, state
    ):
        """Both values are wiring bugs when missing (DelegationPrepareNode always seeds them).

        Without the subsession ID the transcript would be persisted under a key like "supervisor__developer__None",
        colliding across unrelated dispatches; without the call ID the run would be recorded where nothing reads it,
        leaving a delegate_task call unanswered forever.
        """
        with pytest.raises(DelegationFatalError, match="Missing"):
            await dispatch_node.run(state, config={})

        compiled_graph.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graph_interrupt_propagates_unchanged(
        self, dispatch_node, compiled_graph
    ):
        """A subagent's own interrupt() (e.g. tool approval) must bubble up, not be swallowed."""
        compiled_graph.ainvoke.side_effect = GraphInterrupt()

        with pytest.raises(GraphInterrupt):
            await dispatch_node.run(_dispatched_state(1), config={})

    @pytest.mark.asyncio
    async def test_parent_command_propagates_unchanged(
        self, dispatch_node, compiled_graph
    ):
        """A Command(graph=Command.PARENT, ...) bubbling up must also not be swallowed.

        ``ParentCommand`` is a sibling of ``GraphInterrupt`` under LangGraph's
        ``GraphBubbleUp`` base -- not a subclass of it -- so catching only
        ``GraphInterrupt`` would let this fall through to the generic
        ``except Exception`` below and be wrongly recorded as a per-call
        delegation error instead of propagating.
        """
        compiled_graph.ainvoke.side_effect = ParentCommand(
            Command(graph=Command.PARENT)
        )

        with pytest.raises(ParentCommand):
            await dispatch_node.run(_dispatched_state(1), config={})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error,expected_message",
        [
            pytest.param(RuntimeError("boom"), "boom", id="generic-exception"),
            # A stuck subagent is the subagent's own failure, so it stays
            # contained: this is the counterpart to `_FATAL_EXCEPTIONS`, since
            # the supervisor can usefully react to "that subagent looped" by
            # delegating differently.
            pytest.param(
                AgentStuckError("stuck in a loop"),
                "stuck in a loop",
                id="agent-stuck-error",
            ),
        ],
    )
    async def test_contained_exception_is_recorded_as_an_error_run(
        self,
        dispatch_node,
        compiled_graph,
        supervisor_name,
        error,
        expected_message,
    ):
        """A genuine failure must not take down the whole supervisor turn."""
        compiled_graph.ainvoke.side_effect = error

        result = await dispatch_node.run(_dispatched_state(1), config={})

        assert _run_record(result, supervisor_name) == {
            "subsession_id": 1,
            "status": DelegationStatus.ERROR,
            "error": expected_message,
            "final_answer": None,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(GraphRecursionError("limit"), id="graph-recursion-error"),
            pytest.param(InvalidRequestException("bad resume"), id="invalid-request"),
            pytest.param(
                ModelError(
                    error_type=ModelErrorType.AUTHENTICATION_ERROR,
                    status_code=401,
                    message="expired token",
                ),
                id="model-error",
            ),
            pytest.param(
                NotifiableAgentException("Something you should know."),
                id="notifiable-agent-exception",
            ),
            pytest.param(NotifiableException("nope"), id="notifiable-exception"),
            # `PromptInjectionDetectedError` and friends subclass this.
            pytest.param(SecurityException("injection"), id="security-exception"),
            pytest.param(
                InsufficientCredits("out of credits"), id="insufficient-credits"
            ),
            pytest.param(InsufficientEntitlements(), id="usage-quota-error"),
        ],
    )
    async def test_fatal_exceptions_propagate_unchanged(
        self, dispatch_node, compiled_graph, error
    ):
        """Failures that aren't the subagent's to answer for must terminate the flow.

        Containing any of these would lose their distinct downstream handling (a user-facing message, a specific gRPC
        status) and invite the supervisor to re-delegate against something retrying cannot fix.
        """
        compiled_graph.ainvoke.side_effect = error

        with pytest.raises(type(error)):
            await dispatch_node.run(_dispatched_state(1), config={})
