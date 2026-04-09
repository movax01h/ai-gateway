"""Test suite for DelegationNode."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.experimental.components.supervisor.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.nodes.delegation_node import (
    DelegationFatalError,
    DelegationNode,
)
from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys


class TestDelegationNodeNewSession:
    """Tests for DelegationNode creating new sessions."""

    @pytest.fixture(name="delegation_node")
    def delegation_node_fixture(
        self,
        supervisor_name,
        max_delegations,
        delegate_task_cls,
        delegation_count_key,
        active_subsession_key,
        active_subagent_name_key,
        max_subsession_id_key,
        supervisor_history_runtime_key,
        subsession_history_key_factory,
        subsession_goal_key_factory,
    ):
        return DelegationNode(
            name=f"{supervisor_name}#delegation",
            max_delegations=max_delegations,
            delegate_task_cls=delegate_task_cls,
            delegation_count_key=delegation_count_key,
            active_subsession_key=active_subsession_key,
            active_subagent_name_key=active_subagent_name_key,
            max_subsession_id_key=max_subsession_id_key,
            supervisor_history_key=supervisor_history_runtime_key,
            subsession_history_key_factory=subsession_history_key_factory,
            subsession_goal_key_factory=subsession_goal_key_factory,
        )

    @pytest.mark.asyncio
    async def test_new_session_assigns_id_1(
        self,
        delegation_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        ai_message_with_delegate,
    ):
        """Test that a new session gets ID 1 when no prior sessions exist."""
        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [ai_message_with_delegate]

        result = await delegation_node.run(state)

        ctx = result[FlowStateKeys.CONTEXT]
        supervisor_ctx = ctx[supervisor_name]
        assert supervisor_ctx["max_subsession_id"] == 1
        assert supervisor_ctx["active_subsession"] == 1
        assert supervisor_ctx["active_subagent_name"] == developer_name
        assert supervisor_ctx["delegation_count"] == 1

    @pytest.mark.asyncio
    async def test_new_session_does_not_write_conversation_history(
        self,
        delegation_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        ai_message_with_delegate,
    ):
        """Test that a new session does not write an empty conversation history entry.

        The delegation prompt is written to the subsession-scoped goal IOKey instead of being seeded as a HumanMessage.
        The subagent reads the goal via its prompt inputs (RuntimeIOKey), not from conversation history. No explicit
        empty-list write is needed — the history is implicitly absent until the subagent runs.
        """
        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [ai_message_with_delegate]

        result = await delegation_node.run(state)

        session_key = f"{supervisor_name}__{developer_name}__1"
        # No conversation_history entry is written for a new subsession
        assert session_key not in result.get(FlowStateKeys.CONVERSATION_HISTORY, {})

    @pytest.mark.asyncio
    async def test_new_session_increments_from_existing(
        self,
        delegation_node,
        supervisor_name,
        ai_message_with_delegate,
        base_flow_state,
    ):
        """Test that new subsession ID increments from existing max_subsession_id."""
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 3,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 2,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate],
        }

        result = await delegation_node.run(state)

        ctx = result[FlowStateKeys.CONTEXT][supervisor_name]
        assert ctx["max_subsession_id"] == 4
        assert ctx["active_subsession"] == 4
        assert ctx["delegation_count"] == 3

    @pytest.mark.asyncio
    async def test_new_session_writes_goal_to_subsession_key(
        self,
        delegation_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        ai_message_with_delegate,
    ):
        """Test that a new session writes the delegation prompt to the subsession-scoped goal IOKey."""
        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [ai_message_with_delegate]

        result = await delegation_node.run(state)

        # The goal should be written to context:<supervisor>__<subagent>__<subsession_id>/goal
        goal_key = f"{supervisor_name}__{developer_name}__1"
        ctx = result[FlowStateKeys.CONTEXT]
        assert ctx[goal_key]["goal"] == "Implement the feature"


class TestDelegationNodeResumeSession:
    """Tests for DelegationNode resuming existing sessions."""

    @pytest.fixture(name="delegation_node")
    def delegation_node_fixture(
        self,
        supervisor_name,
        max_delegations,
        delegate_task_cls,
        delegation_count_key,
        active_subsession_key,
        active_subagent_name_key,
        max_subsession_id_key,
        supervisor_history_runtime_key,
        subsession_history_key_factory,
        subsession_goal_key_factory,
    ):
        return DelegationNode(
            name=f"{supervisor_name}#delegation",
            max_delegations=max_delegations,
            delegate_task_cls=delegate_task_cls,
            delegation_count_key=delegation_count_key,
            active_subsession_key=active_subsession_key,
            active_subagent_name_key=active_subagent_name_key,
            max_subsession_id_key=max_subsession_id_key,
            supervisor_history_key=supervisor_history_runtime_key,
            subsession_history_key_factory=subsession_history_key_factory,
            subsession_goal_key_factory=subsession_goal_key_factory,
        )

    @pytest.mark.asyncio
    async def test_resume_appends_human_message(
        self,
        delegation_node,
        supervisor_name,
        developer_name,
        ai_message_with_delegate_resume,
        base_flow_state,
    ):
        """Test that resuming a session appends a HumanMessage to existing history."""
        session_key = f"{supervisor_name}__{developer_name}__1"
        existing_history = [HumanMessage(content="Original task")]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 1,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 0,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate_resume],
            session_key: existing_history,
        }

        result = await delegation_node.run(state)

        history = result[FlowStateKeys.CONVERSATION_HISTORY][session_key]
        assert len(history) == 2
        assert history[0].content == "Original task"
        assert isinstance(history[1], HumanMessage)
        assert history[1].content == "Fix the bug in the implementation"

    @pytest.mark.asyncio
    async def test_resume_does_not_write_goal_key(
        self,
        delegation_node,
        supervisor_name,
        developer_name,
        ai_message_with_delegate_resume,
        base_flow_state,
    ):
        """Test that resuming a session does NOT write the goal to the subsession-scoped goal IOKey.

        For resumed sessions, the delegation prompt is appended as a HumanMessage to the conversation history instead.
        """
        session_key = f"{supervisor_name}__{developer_name}__1"
        goal_key = f"{supervisor_name}__{developer_name}__1"

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 1,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 0,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate_resume],
            session_key: [HumanMessage(content="Original task")],
        }

        result = await delegation_node.run(state)

        # Goal key should NOT be written for resumed sessions
        ctx = result.get(FlowStateKeys.CONTEXT, {})
        assert goal_key not in ctx

    @pytest.mark.asyncio
    async def test_resume_does_not_increment_max_id(
        self,
        delegation_node,
        supervisor_name,
        developer_name,
        ai_message_with_delegate_resume,
        base_flow_state,
    ):
        """Test that resuming a session does not increment max_subsession_id."""
        session_key = f"{supervisor_name}__{developer_name}__1"

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 1,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 0,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate_resume],
            session_key: [HumanMessage(content="Original task")],
        }

        result = await delegation_node.run(state)

        ctx = result[FlowStateKeys.CONTEXT][supervisor_name]
        assert ctx["max_subsession_id"] == 1

    @pytest.mark.asyncio
    async def test_resume_invalid_subsession_id_too_high(
        self,
        delegation_node,
        supervisor_name,
        developer_name,
        base_flow_state,
        delegate_tool_call_id,
    ):
        """Test that resuming with an ID higher than max returns error ToolMessage."""
        tool_call = {
            "id": delegate_tool_call_id,
            "name": DelegateTask.tool_title,
            "args": {
                "subagent_name": developer_name,
                "subsession_id": 5,
                "prompt": "Resume",
            },
        }
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [tool_call]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 1,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 0,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_msg],
        }

        result = await delegation_node.run(state)

        # Should return error ToolMessage in supervisor history
        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        assert len(history) == 2  # original + error
        assert isinstance(history[-1], ToolMessage)
        assert "Invalid subsession ID 5" in history[-1].content

    @pytest.mark.asyncio
    async def test_resume_invalid_subsession_id_zero(
        self,
        delegation_node,
        supervisor_name,
        developer_name,
        base_flow_state,
        delegate_tool_call_id,
    ):
        """Test that resuming with subsession ID 0 returns error ToolMessage."""
        tool_call = {
            "id": delegate_tool_call_id,
            "name": DelegateTask.tool_title,
            "args": {
                "subagent_name": developer_name,
                "subsession_id": 0,
                "prompt": "Resume",
            },
        }
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [tool_call]

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 1,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 0,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_msg],
        }

        result = await delegation_node.run(state)

        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        assert isinstance(history[-1], ToolMessage)
        assert "Invalid subsession ID 0" in history[-1].content

    @pytest.mark.asyncio
    async def test_resume_nonexistent_history_returns_error(
        self,
        delegation_node,
        supervisor_name,
        ai_message_with_delegate_resume,
        base_flow_state,
    ):
        """Test that resuming a session with no history returns error ToolMessage."""
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 1,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 0,
            }
        }
        # No session history exists for subsession 1
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate_resume],
        }

        result = await delegation_node.run(state)

        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        assert isinstance(history[-1], ToolMessage)
        assert "No conversation history found for subsession 1" in history[-1].content


class TestDelegationNodeErrorHandling:
    """Tests for DelegationNode error handling."""

    @pytest.fixture(name="delegation_node")
    def delegation_node_fixture(
        self,
        supervisor_name,
        max_delegations,
        delegate_task_cls,
        delegation_count_key,
        active_subsession_key,
        active_subagent_name_key,
        max_subsession_id_key,
        supervisor_history_runtime_key,
        subsession_history_key_factory,
        subsession_goal_key_factory,
    ):
        return DelegationNode(
            name=f"{supervisor_name}#delegation",
            max_delegations=max_delegations,
            delegate_task_cls=delegate_task_cls,
            delegation_count_key=delegation_count_key,
            active_subsession_key=active_subsession_key,
            active_subagent_name_key=active_subagent_name_key,
            max_subsession_id_key=max_subsession_id_key,
            supervisor_history_key=supervisor_history_runtime_key,
            subsession_history_key_factory=subsession_history_key_factory,
            subsession_goal_key_factory=subsession_goal_key_factory,
        )

    @pytest.mark.asyncio
    async def test_max_delegations_reached(
        self,
        supervisor_name,
        delegate_task_cls,
        ai_message_with_delegate,
        base_flow_state,
        delegation_count_key,
        active_subsession_key,
        active_subagent_name_key,
        max_subsession_id_key,
        supervisor_history_runtime_key,
        subsession_history_key_factory,
        subsession_goal_key_factory,
    ):
        """Test that exceeding max_delegations returns error ToolMessage."""
        node = DelegationNode(
            name=f"{supervisor_name}#delegation",
            max_delegations=2,
            delegate_task_cls=delegate_task_cls,
            delegation_count_key=delegation_count_key,
            active_subsession_key=active_subsession_key,
            active_subagent_name_key=active_subagent_name_key,
            max_subsession_id_key=max_subsession_id_key,
            supervisor_history_key=supervisor_history_runtime_key,
            subsession_history_key_factory=subsession_history_key_factory,
            subsession_goal_key_factory=subsession_goal_key_factory,
        )

        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 2,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 2,  # Already at max
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate],
        }

        result = await node.run(state)

        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        assert isinstance(history[-1], ToolMessage)
        assert "Maximum delegation limit" in history[-1].content
        assert "final_response_tool" in history[-1].content

    @pytest.mark.asyncio
    async def test_none_max_delegations_does_not_enforce_limit(
        self,
        supervisor_name,
        developer_name,
        delegate_task_cls,
        ai_message_with_delegate,
        base_flow_state,
        delegation_count_key,
        active_subsession_key,
        active_subagent_name_key,
        max_subsession_id_key,
        supervisor_history_runtime_key,
        subsession_history_key_factory,
        subsession_goal_key_factory,
    ):
        """Test that None max_delegations imposes no delegation limit."""
        node = DelegationNode(
            name=f"{supervisor_name}#delegation",
            max_delegations=None,
            delegate_task_cls=delegate_task_cls,
            delegation_count_key=delegation_count_key,
            active_subsession_key=active_subsession_key,
            active_subagent_name_key=active_subagent_name_key,
            max_subsession_id_key=max_subsession_id_key,
            supervisor_history_key=supervisor_history_runtime_key,
            subsession_history_key_factory=subsession_history_key_factory,
            subsession_goal_key_factory=subsession_goal_key_factory,
        )

        # Set delegation_count to a very high value — should not trigger any limit
        state = {**base_flow_state}
        state["context"] = {
            supervisor_name: {
                "max_subsession_id": 0,
                "active_subsession": None,
                "active_subagent_name": None,
                "delegation_count": 9999,
            }
        }
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_delegate],
        }

        result = await node.run(state)

        # Verify delegation actually happened by asserting on the success-path outputs.
        # The context must reflect the incremented delegation_count, the newly assigned
        # active_subagent_name, and the new subsession ID — none of which are set when
        # the node returns an error ToolMessage instead of delegating.
        ctx = result[FlowStateKeys.CONTEXT][supervisor_name]
        assert ctx["delegation_count"] == 10000  # 9999 + 1
        assert ctx["active_subagent_name"] == developer_name
        assert ctx["active_subsession"] == 1
        assert ctx["max_subsession_id"] == 1

        delegation_goal = subsession_goal_key_factory(
            developer_name, 1
        ).value_from_state(result)
        assert delegation_goal == "Implement the feature"

    @pytest.mark.asyncio
    async def test_no_conversation_history_raises(
        self, delegation_node, base_flow_state
    ):
        """Test that missing supervisor conversation history raises DelegationFatalError."""
        state = {**base_flow_state}
        state["conversation_history"] = {}

        with pytest.raises(DelegationFatalError, match="No conversation history found"):
            await delegation_node.run(state)

    @pytest.mark.asyncio
    async def test_last_message_not_ai_message_raises(
        self, delegation_node, base_flow_state, supervisor_name
    ):
        """Test that non-AIMessage last message raises DelegationFatalError."""
        state = {**base_flow_state}
        state["conversation_history"] = {
            supervisor_name: [HumanMessage(content="Hello")]
        }

        with pytest.raises(DelegationFatalError, match="is not AIMessage"):
            await delegation_node.run(state)

    @pytest.mark.asyncio
    async def test_no_delegate_task_call_raises(
        self, delegation_node, base_flow_state, supervisor_name
    ):
        """Test that missing delegate_task tool call raises DelegationFatalError."""
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"id": "call_1", "name": "read_file", "args": {"file_path": "test.py"}}
        ]

        state = {**base_flow_state}
        state["conversation_history"] = {supervisor_name: [ai_msg]}

        with pytest.raises(
            DelegationFatalError, match="No delegate_task tool call found"
        ):
            await delegation_node.run(state)

    @pytest.mark.asyncio
    async def test_multiple_delegate_calls_returns_error_per_call(
        self,
        delegation_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call_id,
    ):
        """Test that multiple delegate_task calls return one ToolMessage per call."""
        tool_call_1 = {
            "id": delegate_tool_call_id,
            "name": DelegateTask.tool_title,
            "args": {"subagent_name": developer_name, "prompt": "Task A"},
        }
        tool_call_2 = {
            "id": "second_call_id",
            "name": DelegateTask.tool_title,
            "args": {"subagent_name": developer_name, "prompt": "Task B"},
        }
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [tool_call_1, tool_call_2]

        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [ai_msg]

        result = await delegation_node.run(state)

        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        # Original AIMessage + one ToolMessage per delegate_task call
        tool_messages = [m for m in history if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        assert tool_messages[0].tool_call_id == delegate_tool_call_id
        assert tool_messages[1].tool_call_id == "second_call_id"
        assert "Parallel delegation is not supported" in tool_messages[0].content
        assert tool_messages[0].content == tool_messages[1].content

    @pytest.mark.asyncio
    async def test_mixed_tool_calls_returns_error_per_call(
        self,
        delegation_node,
        supervisor_flow_state,
        supervisor_name,
        developer_name,
        delegate_tool_call_id,
    ):
        """Test that mixing delegate_task with other tool calls returns one ToolMessage per call."""
        delegate_call = {
            "id": delegate_tool_call_id,
            "name": DelegateTask.tool_title,
            "args": {"subagent_name": developer_name, "prompt": "Task A"},
        }
        other_call = {
            "id": "other_call_id",
            "name": "read_file",
            "args": {"file_path": "test.py"},
        }
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [delegate_call, other_call]

        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [ai_msg]

        result = await delegation_node.run(state)

        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        tool_messages = [m for m in history if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        assert {tm.tool_call_id for tm in tool_messages} == {
            delegate_tool_call_id,
            "other_call_id",
        }
        assert all("mixed" in tm.content for tm in tool_messages)

    @pytest.mark.asyncio
    async def test_invalid_subagent_name_returns_error(
        self,
        delegation_node,
        supervisor_flow_state,
        supervisor_name,
        delegate_tool_call_id,
    ):
        """Test that an invalid subagent_name is caught by Pydantic validation."""
        tool_call = {
            "id": delegate_tool_call_id,
            "name": DelegateTask.tool_title,
            "args": {
                "subagent_name": "nonexistent",
                "subsession_id": None,
                "prompt": "Do work",
            },
        }
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [tool_call]

        state = supervisor_flow_state
        state["conversation_history"][supervisor_name] = [ai_msg]

        result = await delegation_node.run(state)

        # Pydantic enum validation rejects invalid subagent_name,
        # caught by the try/except and returned as error ToolMessage
        history = result[FlowStateKeys.CONVERSATION_HISTORY][supervisor_name]
        assert isinstance(history[-1], ToolMessage)
        assert "Invalid delegate_task arguments" in history[-1].content
