"""Test suite for DelegationCollectNode."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.components.supervisor_v2.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_collect_node import (
    DelegationCollectNode,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DelegationFatalError,
    DelegationStatus,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys, IOKey


@pytest.fixture(name="collect_node")
def collect_node_fixture(
    supervisor_name,
    delegate_task_cls,
    subsession_run_key_factory,
    supervisor_history_runtime_key,
    ui_history,
):
    return DelegationCollectNode(
        name=f"{supervisor_name}#delegation_collect",
        delegate_task_cls=delegate_task_cls,
        subsession_run_key_factory=subsession_run_key_factory,
        supervisor_history_key=supervisor_history_runtime_key,
        ui_history=ui_history,
    )


def _delegate_call(
    call_id="c1",
    subagent_name="developer",
    subsession_id=None,
    description="Implement the feature",
):
    return {
        "id": call_id,
        "name": DelegateTask.tool_title,
        "args": {
            "subagent_name": subagent_name,
            "description": description,
            "subsession_id": subsession_id,
            "prompt": "Implement the feature",
        },
    }


def _turn(*tool_calls) -> AIMessage:
    """Build the supervisor's last turn, as an AIMessage carrying ``tool_calls``."""
    return AIMessage(content="", tool_calls=list(tool_calls))


def _state_with(state, supervisor_name, history, runs):
    """Seed ``state`` with the supervisor's history and this turn's run records."""
    state["conversation_history"][supervisor_name] = history
    state["context"][supervisor_name]["subsession_runs"] = runs
    return state


class TestDelegationCollectNodeRunOutcome:
    """Tests for DelegationCollectNode.run turning run records into ToolMessages."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "record_kwargs,expected_status,expected_content",
        [
            pytest.param(
                {"final_answer": "Implementation complete."},
                "completed",
                "Implementation complete.",
                id="completed-with-answer",
            ),
            pytest.param(
                {"status": DelegationStatus.ERROR, "error": "boom"},
                "error",
                "failed: boom",
                id="failed-run",
            ),
            pytest.param(
                {"final_answer": None},
                "error",
                "did not produce a final_answer",
                id="completed-without-an-answer",
            ),
        ],
    )
    async def test_run_record_becomes_that_calls_tool_message(
        self,
        collect_node,
        supervisor_flow_state,
        supervisor_name,
        make_run_record,
        record_kwargs,
        expected_status,
        expected_content,
    ):
        state = _state_with(
            supervisor_flow_state,
            supervisor_name,
            [_turn(_delegate_call())],
            {"c1": make_run_record(**record_kwargs)},
        )

        result = await collect_node.run(state, config={})

        tool_message = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][-1]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_call_id == "c1"
        assert f"<status>{expected_status}</status>" in tool_message.content
        assert expected_content in tool_message.content

    @pytest.mark.asyncio
    async def test_reported_subsession_id_comes_from_the_record(
        self,
        collect_node,
        ui_history,
        supervisor_flow_state,
        supervisor_name,
        make_run_record,
    ):
        """The subsession a call ran is the dispatch's to report, not the call's arguments'.

        Only the node that ran the dispatch knows which subsession was minted for it, and the UI needs that ID to group
        the delegation's activity. It is deliberately kept out of the ToolMessage the supervisor's LLM reads.
        """
        state = _state_with(
            supervisor_flow_state,
            supervisor_name,
            [_turn(_delegate_call())],
            {"c1": make_run_record(subsession_id=3)},
        )

        result = await collect_node.run(state, config={})

        assert ui_history.log.success.call_args.kwargs["subsession_id"] == "3"
        tool_message = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][-1]
        assert "subsession_id" not in tool_message.content

    @pytest.mark.asyncio
    async def test_record_is_read_from_wherever_the_factory_points(
        self,
        supervisor_name,
        delegate_task_cls,
        supervisor_history_runtime_key,
        ui_history,
        supervisor_flow_state,
        make_run_record,
    ):
        """The node reads a record through the factory's key, not a convention of its own.

        ``SubagentDispatchNode`` writes through the same factory, so a reader
        that reproduced the naming convention locally could silently drift from
        where the record was actually written. A deliberately unconventional
        factory proves this node spells no key itself.
        """
        node = DelegationCollectNode(
            name=f"{supervisor_name}#delegation_collect",
            delegate_task_cls=delegate_task_cls,
            subsession_run_key_factory=lambda call_id: IOKey(
                target="context", subkeys=["runs", call_id], optional=True
            ),
            supervisor_history_key=supervisor_history_runtime_key,
            ui_history=ui_history,
        )
        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [_turn(_delegate_call())]
        state["context"]["runs"] = {"c1": make_run_record(final_answer="Done.")}

        result = await node.run(state, config={})

        tool_message = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][-1]
        assert "<status>completed</status>" in tool_message.content
        assert "Done." in tool_message.content

    @pytest.mark.asyncio
    async def test_every_dispatched_call_is_answered_in_original_order(
        self,
        collect_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
        make_run_record,
    ):
        state = _state_with(
            supervisor_flow_state,
            supervisor_name,
            [
                _turn(
                    _delegate_call(call_id="c1", subagent_name=developer_name),
                    _delegate_call(call_id="c2", subagent_name=tester_name),
                )
            ],
            {
                "c1": make_run_record(subsession_id=1, final_answer="Dev done."),
                "c2": make_run_record(subsession_id=2, final_answer="Tests done."),
            },
        )

        result = await collect_node.run(state, config={})

        new_messages = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][1:]
        assert [m.tool_call_id for m in new_messages] == ["c1", "c2"]
        assert "Dev done." in new_messages[0].content
        assert "Tests done." in new_messages[1].content

    @pytest.mark.asyncio
    async def test_calls_that_are_already_answered_are_left_alone(
        self,
        collect_node,
        supervisor_flow_state,
        supervisor_name,
        make_run_record,
    ):
        """Only the calls this turn actually dispatched are this node's to answer.

        ``DelegationPrepareNode`` answers every call that failed validation
        itself, and never dispatches it — so an already-answered call has no run
        record to read, and answering it twice would leave the supervisor's
        history with two ToolMessages for one tool call.
        """
        answered_call = _delegate_call(call_id="invalid", subagent_name="developer")
        dispatched_call = _delegate_call(call_id="c1", subagent_name="developer")
        state = _state_with(
            supervisor_flow_state,
            supervisor_name,
            [
                _turn(answered_call, dispatched_call),
                ToolMessage(content="Invalid arguments", tool_call_id="invalid"),
            ],
            {"c1": make_run_record(final_answer="Done.")},
        )

        result = await collect_node.run(state, config={})

        new_messages = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][2:]
        assert [m.tool_call_id for m in new_messages] == ["c1"]

    @pytest.mark.asyncio
    async def test_non_delegate_tool_calls_are_not_answered(
        self,
        collect_node,
        supervisor_flow_state,
        supervisor_name,
        make_run_record,
        regular_tool_call,
    ):
        """Answering another tool's call here would put a delegation result in its place."""
        state = _state_with(
            supervisor_flow_state,
            supervisor_name,
            [_turn(_delegate_call(), regular_tool_call)],
            {"c1": make_run_record(final_answer="Done.")},
        )

        result = await collect_node.run(state, config={})

        new_messages = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][1:]
        assert [m.tool_call_id for m in new_messages] == ["c1"]


class TestDelegationCollectNodeFatalErrors:
    """Tests for the wiring/state bugs DelegationCollectNode refuses to paper over."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "history",
        [
            pytest.param([HumanMessage(content="do it")], id="no-turn-to-answer"),
            pytest.param(
                [
                    _turn(_delegate_call()),
                    ToolMessage(content="already answered", tool_call_id="c1"),
                ],
                id="every-call-already-answered",
            ),
        ],
    )
    async def test_nothing_to_answer_raises_fatal_error(
        self, collect_node, supervisor_flow_state, supervisor_name, history
    ):
        """This node only runs on dispatches, so having nothing to answer is a wiring bug."""
        state = _state_with(supervisor_flow_state, supervisor_name, history, {})

        with pytest.raises(DelegationFatalError, match="No unanswered"):
            await collect_node.run(state, config={})

    @pytest.mark.asyncio
    async def test_missing_run_record_raises_fatal_error(
        self, collect_node, supervisor_flow_state, supervisor_name
    ):
        """A dispatched call with no record is unanswerable, and silence would wedge the next turn.

        ``SubagentDispatchNode`` records every dispatch it completes, failures
        included, so a missing record means the ``Send`` never ran (or ran
        against a different call ID).
        """
        state = _state_with(
            supervisor_flow_state, supervisor_name, [_turn(_delegate_call())], {}
        )

        with pytest.raises(DelegationFatalError, match="No subsession run recorded"):
            await collect_node.run(state, config={})


class TestDelegationCollectNodeUILog:
    """Tests for the UI log entries DelegationCollectNode emits."""

    @pytest.mark.asyncio
    async def test_returns_log_entry_is_tagged_with_the_calls_tool_call_id(
        self,
        collect_node,
        ui_history,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
        make_run_record,
    ):
        """Each result entry must carry its own call's tool_call_id as message_id.

        It's the only correlator a client can use to pair a result with the
        delegation that produced it (and with the ToolMessage answering that
        call) when several delegations are dispatched concurrently in one turn
        and resolve in arbitrary order. Without it,
        ``DefaultUILogWriter._build_log_entry`` falls back to a random UUID.
        """
        state = _state_with(
            supervisor_flow_state,
            supervisor_name,
            [
                _turn(
                    _delegate_call(call_id="c1", subagent_name=developer_name),
                    _delegate_call(call_id="c2", subagent_name=tester_name),
                )
            ],
            {
                "c1": make_run_record(subsession_id=1),
                "c2": make_run_record(subsession_id=2),
            },
        )

        await collect_node.run(state, config={})

        message_ids = [
            call.kwargs["message_id"] for call in ui_history.log.success.call_args_list
        ]
        assert message_ids == ["c1", "c2"]
