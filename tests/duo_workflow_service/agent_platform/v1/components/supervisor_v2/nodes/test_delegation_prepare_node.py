"""Test suite for DelegationPrepareNode."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Send

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    SUBSESSION_ID_CONTEXT_KEY,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_prepare_node import (
    DelegationPrepareNode,
    format_subagent_title,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DELEGATION_CALL_ID_CONTEXT_KEY,
    DelegationFatalError,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys


async def _dispatch(prepare_node, state):
    """Run the node and unpack the ``Command`` it dispatches with.

    Returns ``(update, goto)``: the state update LangGraph would apply, and
    either the list of ``Send`` tasks the node dispatched or the name of the
    node it routed to instead.
    """
    command = await prepare_node.run(state, config={})
    return command.update or {}, command.goto


def _delegate_call(
    call_id: str,
    subagent_name: str,
    prompt: str,
    **extra_args,
) -> dict:
    """Build one ``delegate_task`` tool call, as the supervisor's LLM would emit it."""
    return {
        "id": call_id,
        "name": "delegate_task",
        "args": {
            "subagent_name": subagent_name,
            "description": "Do the thing",
            "prompt": prompt,
            **extra_args,
        },
    }


def _turn_with(state, supervisor_name, *tool_calls):
    """Put a supervisor turn making ``tool_calls`` at the head of ``state``."""
    ai_msg = Mock(spec=AIMessage)
    ai_msg.tool_calls = list(tool_calls)
    state["conversation_history"][supervisor_name] = [ai_msg]
    return state


def _answers(update, supervisor_name) -> list[ToolMessage]:
    """Return the error ToolMessages ``update`` answers this turn's calls with."""
    history = update.get(FlowStateKeys.CONVERSATION_HISTORY, {}).get(
        supervisor_name, []
    )
    return [message for message in history if isinstance(message, ToolMessage)]


class TestFormatSubagentTitle:
    """Tests for format_subagent_title."""

    def test_title_cases_and_joins_with_description(self):
        title = format_subagent_title("research_agent", "Investigate flaky test")
        assert title == "Research Agent Task — Investigate flaky test"

    def test_single_word_name(self):
        title = format_subagent_title("developer", "Fix the bug")
        assert title == "Developer Task — Fix the bug"


@pytest.fixture(name="prepare_node")
def prepare_node_fixture(
    supervisor_name,
    max_delegations,
    delegate_task_cls,
    delegation_count_key,
    max_subsession_id_key,
    supervisor_history_runtime_key,
    ui_history,
):
    return DelegationPrepareNode(
        name=f"{supervisor_name}#delegation_prepare",
        supervisor_name=supervisor_name,
        max_delegations=max_delegations,
        delegate_task_cls=delegate_task_cls,
        delegation_count_key=delegation_count_key,
        max_subsession_id_key=max_subsession_id_key,
        supervisor_history_key=supervisor_history_runtime_key,
        ui_history=ui_history,
    )


class TestDelegationPrepareNodeNewDispatch:
    """Tests for DelegationPrepareNode.run preparing new-subsession dispatches."""

    @pytest.mark.asyncio
    async def test_new_dispatch_builds_isolated_initial_state(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
    ):
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
        )

        _, sends = await _dispatch(prepare_node, state)

        initial_state = sends[0].arg
        assert initial_state["context"]["goal"] == "Implement the feature"
        assert initial_state["context"][SUBSESSION_ID_CONTEXT_KEY] == 1
        assert initial_state["context"][DELEGATION_CALL_ID_CONTEXT_KEY] == "c1"
        assert initial_state["conversation_history"] == {developer_name: []}

    @pytest.mark.asyncio
    async def test_a_stray_subsession_id_argument_still_starts_a_fresh_subsession(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
    ):
        """A model trained on the old schema may still emit subsession_id; it must not resurrect resume.

        Pydantic ignores unknown arguments rather than rejecting them, so the call succeeds. What matters is that it is
        dispatched as a brand-new subsession with an empty transcript, not silently treated as a continuation.
        """
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Keep going", subsession_id=1),
        )

        update, sends = await _dispatch(prepare_node, state)

        initial_state = sends[0].arg
        assert initial_state["conversation_history"] == {developer_name: []}
        assert initial_state["context"]["goal"] == "Keep going"
        assert initial_state["context"][SUBSESSION_ID_CONTEXT_KEY] == 1
        assert update[FlowStateKeys.CONTEXT][supervisor_name]["max_subsession_id"] == 1

    @pytest.mark.asyncio
    async def test_new_dispatch_increments_delegation_count_and_max_id(
        self,
        prepare_node,
        supervisor_name,
        developer_name,
        base_flow_state,
    ):
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {"delegation_count": 2, "max_subsession_id": 3}
        }
        state["conversation_history"] = {}
        _turn_with(
            state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
        )

        update, _ = await _dispatch(prepare_node, state)

        ctx = update[FlowStateKeys.CONTEXT][supervisor_name]
        assert ctx["delegation_count"] == 3
        assert ctx["max_subsession_id"] == 4

    @pytest.mark.asyncio
    async def test_multiple_delegate_calls_in_one_turn_all_dispatch(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
    ):
        """V2 encourages (and dispatches) multiple delegate_task calls in a single turn."""
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
            _delegate_call("c2", tester_name, "Write tests"),
        )

        update, sends = await _dispatch(prepare_node, state)

        assert [
            (send.node, send.arg["context"][SUBSESSION_ID_CONTEXT_KEY])
            for send in sends
        ] == [
            (developer_name, 1),
            (tester_name, 2),
        ]
        assert update[FlowStateKeys.CONTEXT][supervisor_name]["delegation_count"] == 2


class TestDelegationPrepareNodeValidationErrors:
    """Tests for the calls DelegationPrepareNode rejects and answers itself."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_call,delegation_count,expected_content",
        [
            pytest.param(
                _delegate_call("c1", "developer", "Implement the feature"),
                5,
                "Maximum delegation limit",
                id="delegation-limit-reached",
            ),
            pytest.param(
                {
                    "id": "c1",
                    "name": "delegate_task",
                    # Missing description/prompt.
                    "args": {"subagent_name": "developer"},
                },
                0,
                "Invalid delegate_task arguments",
                id="invalid-arguments",
            ),
        ],
    )
    async def test_rejected_call_is_answered_here_and_never_dispatched(
        self,
        prepare_node,
        base_flow_state,
        supervisor_name,
        max_delegations,
        tool_call,
        delegation_count,
        expected_content,
    ):
        """An unvalidatable call runs nothing, so nothing downstream has to know it happened."""
        assert delegation_count in (0, max_delegations)
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "delegation_count": delegation_count,
                "max_subsession_id": 0,
            }
        }
        state["conversation_history"] = {}
        _turn_with(state, supervisor_name, tool_call)

        update, goto = await _dispatch(prepare_node, state)

        assert goto == f"{supervisor_name}#agent"
        answers = _answers(update, supervisor_name)
        assert [message.tool_call_id for message in answers] == ["c1"]
        assert expected_content in answers[0].content

    @pytest.mark.asyncio
    async def test_a_turn_mixing_valid_and_invalid_calls_only_answers_the_invalid_ones(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
    ):
        """The valid calls still dispatch, and stay for DelegationCollectNode to answer.

        Answering a dispatched call here would leave its subagent's result with nowhere to go, and the supervisor with
        two ToolMessages for one call.
        """
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
            {
                "id": "c2",
                "name": "delegate_task",
                "args": {"subagent_name": developer_name},
            },
        )

        update, sends = await _dispatch(prepare_node, state)

        assert [
            send.arg["context"][DELEGATION_CALL_ID_CONTEXT_KEY] for send in sends
        ] == ["c1"]
        assert [
            message.tool_call_id for message in _answers(update, supervisor_name)
        ] == ["c2"]

    @pytest.mark.asyncio
    async def test_mixed_tool_calls_are_rejected_wholesale(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        delegate_tool_call,
        regular_tool_call,
    ):
        """delegate_task mixed with other tools in one turn must be rejected outright."""
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            delegate_tool_call,
            regular_tool_call,
        )

        update, goto = await _dispatch(prepare_node, state)

        assert goto == f"{supervisor_name}#agent"
        answers = _answers(update, supervisor_name)
        assert {message.tool_call_id for message in answers} == {
            delegate_tool_call["id"],
            regular_tool_call["id"],
        }
        assert all("must be the only tool call" in m.content for m in answers)
        # Nothing was dispatched, so there is no bookkeeping to write either.
        assert FlowStateKeys.CONTEXT not in update


class TestDelegationPrepareNodeFatalErrors:
    """Tests for the wiring/state bugs DelegationPrepareNode refuses to paper over."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_calls,match",
        [
            pytest.param(
                [
                    {
                        "id": None,
                        "name": "delegate_task",
                        "args": {
                            "subagent_name": "developer",
                            "description": "Fix bug",
                            "prompt": "Fix the bug",
                        },
                    }
                ],
                "missing an id",
                id="delegate-call-without-an-id",
            ),
            pytest.param(
                [
                    _delegate_call("c1", "developer", "Implement the feature"),
                    {"id": None, "name": "read_file", "args": {}},
                ],
                "missing an id",
                id="mixed-turn-call-without-an-id",
            ),
            pytest.param(
                [{"id": "c1", "name": "read_file", "args": {}}],
                "No delegate_task tool call found",
                id="no-delegate-call-at-all",
            ),
        ],
    )
    async def test_malformed_turn_raises_fatal_error(
        self, prepare_node, supervisor_flow_state, supervisor_name, tool_calls, match
    ):
        state = _turn_with(supervisor_flow_state, supervisor_name, *tool_calls)

        with pytest.raises(DelegationFatalError, match=match):
            await prepare_node.run(state, config={})

    @pytest.mark.asyncio
    async def test_no_conversation_history_raises_fatal_error(
        self, prepare_node, supervisor_flow_state
    ):
        with pytest.raises(DelegationFatalError):
            await prepare_node.run(supervisor_flow_state, config={})

    @pytest.mark.asyncio
    async def test_last_message_not_ai_message_raises_fatal_error(
        self, prepare_node, supervisor_flow_state, supervisor_name
    ):
        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [HumanMessage(content="hello")]

        with pytest.raises(DelegationFatalError):
            await prepare_node.run(state, config={})


class TestDelegationPrepareNodeUILog:
    """Tests for the UI log entries DelegationPrepareNode emits.

    Every entry must be tagged with the tool_call_id of the ``delegate_task``
    call it describes, so a client can pair it with the matching
    ``ON_DELEGATION_RETURNS`` entry (see ``DelegationCollectNode``) and with the
    ToolMessage answering that call. With several delegations dispatched
    concurrently in a single turn, the tool_call_id is the only reliable
    correlator; without it ``DefaultUILogWriter._build_log_entry`` falls back
    to a random UUID.
    """

    @pytest.mark.asyncio
    async def test_dispatch_log_entries_are_tagged_with_their_tool_call_ids(
        self,
        prepare_node,
        ui_history,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
    ):
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
            _delegate_call("c2", tester_name, "Write tests"),
        )

        await prepare_node.run(state, config={})

        message_ids = [
            call.kwargs["message_id"] for call in ui_history.log.success.call_args_list
        ]
        assert message_ids == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_validation_error_log_entry_is_tagged_with_its_tool_call_id(
        self,
        prepare_node,
        ui_history,
        supervisor_name,
        developer_name,
        base_flow_state,
        max_delegations,
    ):
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "delegation_count": max_delegations,
                "max_subsession_id": 0,
            }
        }
        state["conversation_history"] = {}
        _turn_with(
            state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
        )

        await prepare_node.run(state, config={})

        ui_history.log.error.assert_called_once()
        assert ui_history.log.error.call_args.kwargs["message_id"] == "c1"

    @pytest.mark.asyncio
    async def test_mixed_tool_calls_log_entries_are_tagged_per_call(
        self,
        prepare_node,
        ui_history,
        supervisor_flow_state,
        supervisor_name,
        delegate_tool_call,
        regular_tool_call,
    ):
        """The mixed-turn rejection responds to every call, so every entry needs its own id."""
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            delegate_tool_call,
            regular_tool_call,
        )

        await prepare_node.run(state, config={})

        message_ids = {
            call.kwargs["message_id"] for call in ui_history.log.error.call_args_list
        }
        assert message_ids == {delegate_tool_call["id"], regular_tool_call["id"]}


class TestDelegationPrepareNodeDispatch:
    """Tests for the ``Command`` DelegationPrepareNode dispatches with.

    Every valid ``delegate_task`` call becomes a ``Send`` task in the command's
    ``goto``; when there is nothing left to dispatch, ``goto`` names the
    supervisor's agent node directly instead.
    """

    @pytest.mark.asyncio
    async def test_valid_call_is_dispatched_as_a_send(
        self, prepare_node, supervisor_flow_state, supervisor_name, developer_name
    ):
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
        )

        _, sends = await _dispatch(prepare_node, state)

        assert len(sends) == 1
        assert isinstance(sends[0], Send)
        assert sends[0].node == developer_name

    @pytest.mark.asyncio
    async def test_every_valid_call_in_one_turn_is_dispatched_concurrently(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
    ):
        state = _turn_with(
            supervisor_flow_state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
            _delegate_call("c2", tester_name, "Test the feature"),
        )

        _, sends = await _dispatch(prepare_node, state)

        assert {send.node for send in sends} == {developer_name, tester_name}
        assert {send.arg["context"]["goal"] for send in sends} == {
            "Implement the feature",
            "Test the feature",
        }

    @pytest.mark.asyncio
    async def test_payloads_omit_the_supervisors_own_bookkeeping(
        self, prepare_node, supervisor_flow_state, supervisor_name, developer_name
    ):
        """No subagent is handed the supervisor's delegation records.

        Those records hold every earlier delegation's full answer, and each dispatch is checkpointed, so copying them in
        would make a long session's checkpoints grow with the square of its delegation count. The rest of the
        supervisor's context still travels, since that is what the subagent's own prompt inputs read.
        """
        state = supervisor_flow_state
        state["context"][supervisor_name] = {
            "delegation_count": 1,
            "max_subsession_id": 1,
            "subsession_runs": {
                "c0": {
                    "subsession_id": 1,
                    "status": "completed",
                    "error": None,
                    "final_answer": "A very long answer from an earlier delegation.",
                }
            },
        }
        state["context"]["project_id"] = 42
        _turn_with(
            state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
        )

        _, sends = await _dispatch(prepare_node, state)

        assert supervisor_name not in sends[0].arg["context"]
        assert sends[0].arg["context"]["project_id"] == 42

    @pytest.mark.asyncio
    async def test_turn_where_every_call_failed_validation_goes_back_to_the_agent(
        self,
        prepare_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        max_delegations,
    ):
        """Nothing was dispatched and every call is already answered here."""
        state = supervisor_flow_state
        state["context"][supervisor_name] = {
            "delegation_count": max_delegations,
            "max_subsession_id": 0,
        }
        _turn_with(
            state,
            supervisor_name,
            _delegate_call("c1", developer_name, "Implement the feature"),
        )

        update, goto = await _dispatch(prepare_node, state)

        assert goto == f"{supervisor_name}#agent"
        assert [
            message.tool_call_id for message in _answers(update, supervisor_name)
        ] == ["c1"]
