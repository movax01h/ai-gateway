"""Test suite for SubagentReturnNode."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from duo_workflow_service.agent_platform.experimental.components.supervisor.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.nodes.subagent_return_node import (
    SubagentReturnNode,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.experimental.state.base import RuntimeIOKey


@pytest.fixture(name="final_answer_runtime_key")
def final_answer_runtime_key_fixture(supervisor_name):
    """RuntimeIOKey that resolves the subagent's final_answer IOKey from state."""

    def factory(state):
        active_session = (
            state.get("context", {}).get(supervisor_name, {}).get("active_subsession")
        )
        active_type = (
            state.get("context", {})
            .get(supervisor_name, {})
            .get("active_subagent_type")
        )
        if active_session is None or active_type is None:
            return IOKey(
                target="context",
                subkeys=[supervisor_name, "UNKNOWN", "UNKNOWN", "final_answer"],
                optional=True,
            )
        return IOKey(
            target="context",
            subkeys=[supervisor_name, active_type, str(active_session), "final_answer"],
            optional=True,
        )

    return RuntimeIOKey(alias="final_answer", factory=factory)


@pytest.fixture(name="return_node")
def return_node_fixture(
    supervisor_name,
    delegate_task_cls,
    active_subsession_key,
    active_subagent_type_key,
    supervisor_history_runtime_key,
    final_answer_runtime_key,
):
    """SubagentReturnNode wired with supervisor-scoped key fixtures."""
    return SubagentReturnNode(
        name=f"{supervisor_name}#subagent_return",
        delegate_task_cls=delegate_task_cls,
        active_subsession_key=active_subsession_key,
        active_subagent_type_key=active_subagent_type_key,
        final_answer_key=final_answer_runtime_key,
        supervisor_history_key=supervisor_history_runtime_key,
    )


def _state_with_delegate_history(
    base_flow_state, supervisor_name, delegate_tool_call, extra_context=None
):
    """Build flow state with an AIMessage carrying a delegate_task call in supervisor history."""
    ai_msg = Mock(spec=AIMessage)
    ai_msg.tool_calls = [delegate_tool_call]
    state = {**base_flow_state}
    state["context"] = {
        supervisor_name: {
            "max_subsession_id": 1,
            "active_subsession": 1,
            "active_subagent_type": "developer",
            "delegation_count": 1,
            **(extra_context or {}),
        }
    }
    state["conversation_history"] = {supervisor_name: [ai_msg]}
    return state


class TestSubagentReturnNodeRun:
    """Tests for SubagentReturnNode.run — all assertions via the public run() interface."""

    @pytest.mark.asyncio
    async def test_injects_tool_message_on_success(
        self,
        return_node,
        supervisor_state_with_completed_subsession,
        supervisor_name,
        developer_name,
    ):
        """Successful subagent result is injected as a ToolMessage with completed XML."""
        result = await return_node.run(supervisor_state_with_completed_subsession)

        supervisor_history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        tool_msg = supervisor_history[-1]
        assert isinstance(tool_msg, ToolMessage)
        assert "<delegation_result>" in tool_msg.content
        assert f"<subagent_type>{developer_name}</subagent_type>" in tool_msg.content
        assert "<subsession_id>1</subsession_id>" in tool_msg.content
        assert "<status>completed</status>" in tool_msg.content
        assert "<result>" in tool_msg.content
        assert "<error>" not in tool_msg.content
        assert "Implementation complete. Created feature X." in tool_msg.content

    @pytest.mark.asyncio
    async def test_resets_active_subsession(
        self,
        return_node,
        supervisor_state_with_completed_subsession,
        supervisor_name,
    ):
        """Active subsession and subagent type are cleared after return."""
        result = await return_node.run(supervisor_state_with_completed_subsession)

        ctx = result[FlowStateKeys.CONTEXT][supervisor_name]
        assert ctx["active_subsession"] is None
        assert ctx["active_subagent_type"] is None

    @pytest.mark.asyncio
    async def test_tool_call_id_matches_delegate_call(
        self,
        return_node,
        supervisor_state_with_completed_subsession,
        supervisor_name,
        delegate_tool_call_id,
    ):
        """ToolMessage.tool_call_id matches the preceding delegate_task call ID."""
        result = await return_node.run(supervisor_state_with_completed_subsession)

        supervisor_history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        tool_msg = supervisor_history[-1]
        assert tool_msg.tool_call_id == delegate_tool_call_id

    @pytest.mark.asyncio
    async def test_uses_most_recent_delegate_call_id(
        self,
        return_node,
        base_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call,
    ):
        """When history contains multiple AIMessages, the most recent delegate call ID is used."""
        old_call = {
            "id": "old_call_id",
            "name": DelegateTask.tool_title,
            "args": {"subagent_type": developer_name, "prompt": "Old task"},
        }
        old_msg = Mock(spec=AIMessage)
        old_msg.tool_calls = [old_call]
        new_msg = Mock(spec=AIMessage)
        new_msg.tool_calls = [delegate_tool_call]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "active_subsession": 1,
                "active_subagent_type": developer_name,
                developer_name: {"1": {"final_answer": "Done"}},
            }
        }
        state["conversation_history"] = {supervisor_name: [old_msg, new_msg]}

        result = await return_node.run(state)

        tool_msg = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][-1]
        assert tool_msg.tool_call_id == delegate_tool_call["id"]

    @pytest.mark.asyncio
    async def test_error_status_when_no_final_answer(
        self,
        return_node,
        base_flow_state,
        supervisor_name,
        delegate_tool_call,
    ):
        """Missing final_answer produces error XML in the ToolMessage."""
        state = _state_with_delegate_history(
            base_flow_state, supervisor_name, delegate_tool_call
        )

        result = await return_node.run(state)

        tool_msg = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name][-1]
        assert isinstance(tool_msg, ToolMessage)
        assert "<status>error</status>" in tool_msg.content
        assert "<error>" in tool_msg.content
        assert "<result>" not in tool_msg.content

    @pytest.mark.asyncio
    async def test_no_active_session_raises(
        self, return_node, base_flow_state, supervisor_name
    ):
        """Missing active session raises ValueError before any history lookup."""
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {"active_subsession": None, "active_subagent_type": None}
        }

        with pytest.raises(ValueError, match="No active subsession found"):
            await return_node.run(state)

    @pytest.mark.asyncio
    async def test_empty_supervisor_history_raises(
        self, return_node, base_flow_state, supervisor_name, developer_name
    ):
        """Empty supervisor history raises ValueError — no delegate call to respond to."""
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "active_subsession": 1,
                "active_subagent_type": developer_name,
            }
        }
        state["conversation_history"] = {supervisor_name: []}

        with pytest.raises(
            ValueError, match="No conversation history found for supervisor"
        ):
            await return_node.run(state)

    @pytest.mark.asyncio
    async def test_no_delegate_call_in_history_raises(
        self,
        return_node,
        base_flow_state,
        supervisor_name,
        developer_name,
    ):
        """History with no delegate_task call raises ValueError."""
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [{"id": "c1", "name": "read_file", "args": {}}]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "active_subsession": 1,
                "active_subagent_type": developer_name,
            }
        }
        state["conversation_history"] = {supervisor_name: [ai_msg]}

        with pytest.raises(
            ValueError, match="Could not find delegate_task tool_call_id"
        ):
            await return_node.run(state)

    @pytest.mark.asyncio
    async def test_multiple_delegate_calls_in_message_raises(
        self,
        return_node,
        base_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call,
    ):
        """Multiple delegate_task calls in a single AIMessage raises ValueError."""
        second_call = {**delegate_tool_call, "id": "second_call_id"}
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [delegate_tool_call, second_call]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "active_subsession": 1,
                "active_subagent_type": developer_name,
            }
        }
        state["conversation_history"] = {supervisor_name: [ai_msg]}

        with pytest.raises(ValueError, match="2 delegate_task calls"):
            await return_node.run(state)

    @pytest.mark.asyncio
    async def test_delegate_call_mixed_with_other_tools_raises(
        self,
        return_node,
        base_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call,
    ):
        """delegate_task mixed with other tool calls in one AIMessage raises ValueError."""
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [
            delegate_tool_call,
            {"id": "other", "name": "read_file", "args": {}},
        ]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "active_subsession": 1,
                "active_subagent_type": developer_name,
            }
        }
        state["conversation_history"] = {supervisor_name: [ai_msg]}

        with pytest.raises(ValueError, match="mixed with other tool calls"):
            await return_node.run(state)
