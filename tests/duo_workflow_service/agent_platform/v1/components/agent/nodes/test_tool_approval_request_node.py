"""Test suite for v1 ToolApprovalRequestNode class."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_request_node import (
    ToolApprovalRequestNode,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.agent_platform.v1.state.base import (
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.ui_log import (
    UIHistory,
    default_ui_log_writer_class,
)
from duo_workflow_service.entities import (
    ApprovalSource,
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools import (
    MalformedToolCallError,
    Toolset,
    UnknownToolError,
)

_APPROVAL_EVENTS = [UILogEventsAgent.ON_TOOL_APPROVAL_REQUEST]


@pytest.fixture(name="conversation_history_key")
def conversation_history_key_fixture(component_name):
    """Fixture for conversation history key."""
    return IOKey(target="conversation_history", subkeys=[component_name])


@pytest.fixture(name="status_key")
def status_key_fixture():
    """Fixture for status key."""
    return IOKey(target="status")


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    """Fixture for mock toolset."""
    # spec keeps the mock in sync with the real Toolset API: renaming or
    # removing a method on Toolset fails these tests instead of passing vacuously
    mock_toolset = MagicMock(spec=Toolset)

    # Mock tool
    mock_tool = Mock()
    mock_tool.name = "test_tool"

    # Setup toolset to return tool by name
    mock_toolset.__getitem__.return_value = mock_tool

    # Tools are NOT pre-approved or session-approved by default (require approval).
    # resolve_approval_source is async on Toolset, so the spec makes it an
    # AsyncMock; None means "human approval required".
    mock_toolset.resolve_approval_source.return_value = None

    return mock_toolset


@pytest.fixture(name="make_ui_history")
def make_ui_history_fixture(component_name):
    """Build a real history: the writer stamps the fields under test, so a mock would be vacuous."""

    def _make(events=_APPROVAL_EVENTS):
        return UIHistory(
            events=events,
            writer_class=default_ui_log_writer_class(
                events_class=UILogEventsAgent,
                ui_role_as="request",
                component_name=component_name,
            ),
        )

    return _make


@pytest.fixture(name="ui_history")
def ui_history_fixture(make_ui_history):
    """Fixture for a UI history wired as the component wires it."""
    return make_ui_history()


@pytest.fixture(name="session_id_key")
def session_id_key_fixture():
    """Fixture for a subsession-scoped session ID key, as a supervisor would pass."""
    return IOKey(
        target="context",
        subkeys=["supervisor", "active_subsession"],
        optional=True,
    )


@pytest.fixture(name="make_node")
def make_node_fixture(conversation_history_key, status_key, mock_toolset, ui_history):
    """Build a node, overriding only what a test cares about."""

    def _make(pre_approved_tools=(), ui_history=ui_history, **kwargs):
        return ToolApprovalRequestNode(
            name="test_agent#tool_approval_request",
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            toolset=mock_toolset,
            pre_approved_tools=list(pre_approved_tools),
            status_key=RuntimeIOKey(alias="status", factory=lambda _: status_key),
            ui_history=ui_history,
            **kwargs,
        )

    return _make


@pytest.fixture(name="tool_approval_request_node")
def tool_approval_request_node_fixture(make_node):
    """Fixture for ToolApprovalRequestNode instance."""
    return make_node()


@pytest.fixture(name="state_with")
def state_with_fixture(base_flow_state, component_name):
    """Build flow state holding *messages*, optionally under an active subsession."""

    def _state(messages, active_subsession=None):
        return {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {component_name: messages},
            "context": {
                **base_flow_state.get("context", {}),
                "supervisor": {"active_subsession": active_subsession},
            },
        }

    return _state


@pytest.fixture(name="mock_ai_message_with_tool_calls")
def mock_ai_message_with_tool_calls_fixture():
    """Fixture for AIMessage with tool calls."""
    mock_message = Mock(spec=AIMessage)
    mock_message.tool_calls = [
        {"id": "call_123", "name": "test_tool", "args": {"param": "value"}},
    ]
    return mock_message


class TestToolApprovalRequestNodeValidCalls:
    """Test suite for valid tool call handling."""

    @pytest.mark.asyncio
    async def test_valid_tool_calls_creates_ui_logs_and_sets_status(
        self,
        make_node,
        state_with,
        session_id_key,
        component_name,
        mock_ai_message_with_tool_calls,
    ):
        """Test the entry shape field-for-field, since it is the client-facing contract.

        ``additional_context`` is the fragile one: the writer defaults it to
        ``[]`` when the kwarg is omitted, so the node must pass ``None``.
        """
        state = state_with([mock_ai_message_with_tool_calls], active_subsession=2)

        with patch(
            "duo_workflow_service.agent_platform.v1.components."
            "agent.nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.return_value = "Execute test_tool with param=value"

            result = await make_node(session_id_key=session_id_key).run(state)

        assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED.value
        (entry,) = result["ui_chat_log"]
        assert entry == {
            "component_name": component_name,
            "subsession_id": "2",
            # Clients key the approve/reject controls off this.
            "message_type": MessageTypeEnum.REQUEST,
            "content": "Execute test_tool with param=value",
            "status": ToolStatus.SUCCESS,
            # The tool call's own id, so this resolves the streamed PENDING card.
            "message_id": "call_123",
            "tool_info": {"name": "test_tool", "args": {"param": "value"}},
            "additional_context": None,
            "message_sub_type": None,
            "correlation_id": None,
            "timestamp": entry["timestamp"],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("active_subsession", "expected_subsession_id"),
        [
            # A supervisor writes the subsession as an int; nodes stringify it.
            pytest.param(3, "3", id="subagent"),
            # 0 is a real subsession ID, so it must not be read as absent.
            pytest.param(0, "0", id="subagent-zero"),
            pytest.param(None, None, id="standalone"),
        ],
    )
    async def test_every_call_in_a_batch_gets_an_attributed_entry(
        self,
        make_node,
        state_with,
        session_id_key,
        component_name,
        mock_ai_message_with_multiple_tool_calls,
        active_subsession,
        expected_subsession_id,
    ):
        """One entry per call, each attributed, so a stamp on only the first fails."""
        node = make_node(session_id_key=session_id_key)
        state = state_with(
            [mock_ai_message_with_multiple_tool_calls], active_subsession
        )

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.side_effect = ["Display A", "Display B"]

            result = await node.run(state)

        assert [
            (e["message_id"], e["component_name"], e["subsession_id"])
            for e in result["ui_chat_log"]
        ] == [
            ("tool_call_id_1", component_name, expected_subsession_id),
            ("tool_call_id_2", component_name, expected_subsession_id),
        ]

    @pytest.mark.asyncio
    async def test_tool_with_none_display_message_skipped(
        self,
        tool_approval_request_node,
        state_with,
        mock_ai_message_with_multiple_tool_calls,
    ):
        """Test that tools with None display message are skipped from UI logs."""
        state = state_with([mock_ai_message_with_multiple_tool_calls])

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            # First returns None (skip), second returns display text
            mock_format.side_effect = [None, "Display B"]

            result = await tool_approval_request_node.run(state)

            # Should only have 1 UI log (second one)
            assert len(result["ui_chat_log"]) == 1
            assert result["ui_chat_log"][0]["message_id"] == "tool_call_id_2"


class TestToolApprovalRequestNodeInvalidCalls:
    """Test suite for invalid tool call handling."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_invalid_tool_calls_returns_error_messages(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
    ):
        """Test that invalid tool calls return error ToolMessages."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_123", "name": "bad_tool", "args": {"invalid": "args"}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # Make validate_tool_call raise MalformedToolCallError
        error = MalformedToolCallError(
            "Invalid arguments",
            tool_call={"id": "call_123", "name": "bad_tool"},
        )
        mock_toolset.validate_tool_call.side_effect = error

        result = await tool_approval_request_node.run(state)

        # Should include conversation history with error messages
        assert "conversation_history" in result
        assert component_name in result["conversation_history"]

        new_messages = result["conversation_history"][component_name]
        # Should have original message + 1 error ToolMessage
        assert len(new_messages) == 2
        assert isinstance(new_messages[0], AIMessage)
        assert isinstance(new_messages[1], ToolMessage)

        # Verify error message
        assert new_messages[1].tool_call_id == "call_123"
        assert "Invalid arguments" in str(new_messages[1].content)

        # Should set status to EXECUTION (not approval required)
        assert "status" in result
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_mixed_valid_invalid_rejects_entire_batch(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
    ):
        """Test that when any call is invalid, the entire batch is rejected.

        Every tool_call_id in the AIMessage must have a corresponding ToolMessage or Anthropic returns a 400 error.
        Valid calls are cancelled so the LLM can replan from scratch.
        """
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_good", "name": "good_tool", "args": {}},
            {"id": "call_bad", "name": "bad_tool", "args": {}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # First call valid, second call invalid
        def validate_side_effect(tool_call):
            if tool_call["id"] == "call_bad":
                raise MalformedToolCallError("Bad tool error", tool_call=tool_call)

        mock_toolset.validate_tool_call.side_effect = validate_side_effect

        result = await tool_approval_request_node.run(state)

        # Should return a ToolMessage for every call in the batch (original + 2 errors)
        new_messages = result["conversation_history"][component_name]
        assert len(new_messages) == 3
        assert isinstance(new_messages[1], ToolMessage)
        assert isinstance(new_messages[2], ToolMessage)

        # Valid call gets a cancellation message
        assert new_messages[1].tool_call_id == "call_good"
        assert "cancelled" in new_messages[1].content

        # Invalid call gets the actual error
        assert new_messages[2].tool_call_id == "call_bad"
        assert "Bad tool error" in new_messages[2].content


class TestToolApprovalRequestNodeNoToolCalls:
    """Test suite for no tool calls handling."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_no_tool_calls_returns_error_human_message(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that AIMessage without tool calls returns error HumanMessage."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        result = await tool_approval_request_node.run(state)

        # Should include conversation history with error message
        assert "conversation_history" in result
        assert component_name in result["conversation_history"]

        new_messages = result["conversation_history"][component_name]
        # Should have original message + 1 error HumanMessage
        assert len(new_messages) == 2
        assert isinstance(new_messages[1], HumanMessage)
        assert "No tool calls found" in new_messages[1].content

        # Should set status to EXECUTION
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("conversation_history_key")
    async def test_non_ai_message_returns_error(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
    ):
        """Test that non-AIMessage returns error HumanMessage."""
        mock_message = Mock(spec=HumanMessage)
        mock_message.content = "user message"

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        result = await tool_approval_request_node.run(state)

        # Should return error message
        new_messages = result["conversation_history"][component_name]
        assert len(new_messages) == 2
        assert isinstance(new_messages[1], HumanMessage)
        assert "No tool calls found" in new_messages[1].content


class TestToolApprovalRequestNodePreApproved:
    """Test suite for pre-approved tool handling."""

    @pytest.mark.asyncio
    async def test_all_pre_approved_skips_approval(
        self,
        conversation_history_key,
        status_key,
        mock_toolset,
        ui_history,
        base_flow_state,
        component_name,
    ):
        """Test that when all tools are pre-approved, approval is skipped."""
        # Create node with approved tools
        node = ToolApprovalRequestNode(
            name="test_agent#tool_approval_request",
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: conversation_history_key
            ),
            toolset=mock_toolset,
            pre_approved_tools=["approved_tool"],
            status_key=RuntimeIOKey(alias="status", factory=lambda _: status_key),
            ui_history=ui_history,
        )

        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "approved_tool", "args": {}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        # Mock _should_skip_approval to return a source for approved_tool
        with patch.object(
            node,
            "_should_skip_approval",
            return_value=ApprovalSource.PREAPPROVED_CONFIG,
        ):
            result = await node.run(state)

            # Should set status to EXECUTION for explicit routing
            assert "status" in result
            assert result["status"] == WorkflowStatusEnum.EXECUTION.value

            # Should NOT include ui_chat_log
            assert "ui_chat_log" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pre_approved_tools,toolset_source,expected_source",
        [
            (
                ["test_tool"],
                ApprovalSource.SESSION_APPROVAL,
                ApprovalSource.PREAPPROVED_CONFIG,
            ),
            ([], ApprovalSource.PREAPPROVED_CONFIG, ApprovalSource.PREAPPROVED_CONFIG),
            ([], ApprovalSource.SESSION_APPROVAL, ApprovalSource.SESSION_APPROVAL),
            ([], None, None),
            ([], UnknownToolError("unknown"), None),
        ],
        ids=[
            "component_pre_approved_short_circuits_toolset",
            "toolset_privilege_pre_approved",
            "toolset_session_approved",
            "not_pre_approved",
            "unknown_tool_defers_to_validation",
        ],
    )
    async def test_approval_skip_source_resolution_through_run(
        self,
        make_node,
        state_with,
        mock_toolset,
        mock_ai_message_with_tool_calls,
        pre_approved_tools,
        toolset_source,
        expected_source,
    ):
        """Run() skips approval per mechanism and logs the resolved source."""
        if isinstance(toolset_source, Exception):
            mock_toolset.resolve_approval_source.side_effect = toolset_source
        else:
            mock_toolset.resolve_approval_source.return_value = toolset_source

        node = make_node(pre_approved_tools=pre_approved_tools)
        state = state_with([mock_ai_message_with_tool_calls])

        with (
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent."
                "nodes.tool_approval_request_node.log"
            ) as mock_log,
            patch(
                "duo_workflow_service.agent_platform.v1.components.agent."
                "nodes.tool_approval_request_node.format_tool_display_message"
            ) as mock_format,
        ):
            mock_format.return_value = "Execute test_tool"

            result = await node.run(state)

        if expected_source is None:
            assert (
                result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED.value
            )
            assert "ui_chat_log" in result
            mock_log.info.assert_not_called()
        else:
            assert result["status"] == WorkflowStatusEnum.EXECUTION.value
            assert "ui_chat_log" not in result
            # The skip is logged with the mechanism that granted it.
            mock_log.info.assert_called_once()
            assert mock_log.info.call_args.kwargs["approval_source"] == expected_source

        # Component-level pre-approval must short-circuit before any toolset check.
        if pre_approved_tools:
            mock_toolset.resolve_approval_source.assert_not_awaited()


class TestToolApprovalRequestNodeSessionApprovals:
    """Test suite for session-approved tool call handling."""

    @pytest.mark.asyncio
    async def test_session_approved_call_skips_approval(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
        mock_ai_message_with_tool_calls,
    ):
        """Test that a session-approved tool call skips approval entirely."""
        mock_toolset.resolve_approval_source.return_value = (
            ApprovalSource.SESSION_APPROVAL
        )

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.log"
        ) as mock_log:
            result = await tool_approval_request_node.run(state)

        mock_toolset.resolve_approval_source.assert_awaited_once_with(
            "test_tool", {"param": "value"}
        )
        # The skip is logged with the session-approval provenance.
        mock_log.info.assert_called_once()
        assert (
            mock_log.info.call_args.kwargs["approval_source"]
            == ApprovalSource.SESSION_APPROVAL
        )
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value
        assert "ui_chat_log" not in result

    @pytest.mark.asyncio
    async def test_mixed_batch_prompts_only_for_unapproved_calls(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
    ):
        """Test that same-tool calls with different args are evaluated independently."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "test_tool", "args": {"cmd": "approved"}},
            {"id": "call_2", "name": "test_tool", "args": {"cmd": "unapproved"}},
        ]

        # Only the first call is session-approved
        mock_toolset.resolve_approval_source.side_effect = lambda _name, args: (
            ApprovalSource.SESSION_APPROVAL if args == {"cmd": "approved"} else None
        )

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.return_value = "Execute test_tool"

            result = await tool_approval_request_node.run(state)

        assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED.value
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_id"] == "call_2"
        assert mock_toolset.resolve_approval_source.await_count == 2

    @pytest.mark.asyncio
    async def test_component_pre_approved_tools_skip_toolset_check(
        self,
        make_node,
        mock_toolset,
        base_flow_state,
        component_name,
    ):
        """Test that component-level pre-approved tools never consult the toolset."""
        node = make_node(pre_approved_tools=["approved_tool"])

        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {"id": "call_1", "name": "approved_tool", "args": {}},
        ]

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: [mock_message]}

        result = await node.run(state)

        mock_toolset.resolve_approval_source.assert_not_awaited()
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value
        assert "ui_chat_log" not in result

    @pytest.mark.asyncio
    async def test_unknown_tool_in_approval_check_still_requires_approval(
        self,
        tool_approval_request_node,
        base_flow_state,
        component_name,
        mock_toolset,
        mock_ai_message_with_tool_calls,
    ):
        """Test the defensive UnknownToolError path.

        Unknown tools are rejected by _filter_tool_calls before the approval check runs, so this path should be
        unreachable; if it is ever hit, the call must fail toward requiring approval.
        """
        mock_toolset.resolve_approval_source.side_effect = UnknownToolError("nope")

        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {
            component_name: [mock_ai_message_with_tool_calls]
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.return_value = "Execute test_tool"

            result = await tool_approval_request_node.run(state)

        assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED.value


class TestToolApprovalRequestNodeEdgeCases:
    """Test suite for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_history_raises_error(
        self, tool_approval_request_node, base_flow_state, component_name
    ):
        """Test that empty conversation history raises RuntimeError."""
        state = base_flow_state.copy()
        state[FlowStateKeys.CONVERSATION_HISTORY] = {component_name: []}

        with pytest.raises(RuntimeError, match="No conversation history found"):
            await tool_approval_request_node.run(state)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rendered", "events"),
        [
            pytest.param(None, _APPROVAL_EVENTS, id="nothing-renderable"),
            # An event filter dropping the prompt is what a count of written
            # entries cannot see. Both must raise rather than report
            # TOOL_CALL_APPROVAL_REQUIRED with an empty log.
            pytest.param(
                "Execute tool_a",
                [UILogEventsAgent.ON_AGENT_FINAL_ANSWER],
                id="event-filtered-out",
            ),
        ],
    )
    async def test_raises_when_no_prompt_reaches_the_chat_log(
        self,
        make_node,
        make_ui_history,
        state_with,
        mock_ai_message_with_tool_calls,
        rendered,
        events,
    ):
        """Test that a flow is never left waiting on a prompt the user never saw."""
        node = make_node(ui_history=make_ui_history(events))

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.return_value = rendered

            with pytest.raises(
                RuntimeError, match="No valid tool calls found to display for approval"
            ):
                await node.run(state_with([mock_ai_message_with_tool_calls]))

    @pytest.mark.asyncio
    async def test_a_failed_render_does_not_leak_into_the_next_run(
        self,
        tool_approval_request_node,
        state_with,
        mock_ai_message_with_multiple_tool_calls,
    ):
        """All rendering happens before the first write.

        ``UIHistory`` only clears in ``pop_state_updates``, so writing entries as
        they render would strand them for a later run to flush as its own.
        """
        message = mock_ai_message_with_multiple_tool_calls
        state = state_with([message])

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent."
            "nodes.tool_approval_request_node.format_tool_display_message"
        ) as mock_format:
            mock_format.side_effect = ["Display A", RuntimeError("render blew up")]
            with pytest.raises(RuntimeError, match="render blew up"):
                await tool_approval_request_node.run(state)

            mock_format.side_effect = None
            mock_format.return_value = "Display C"
            message.tool_calls = [{"id": "call_3", "name": "tool_c", "args": {"z": 3}}]

            result = await tool_approval_request_node.run(state)

        assert [entry["message_id"] for entry in result["ui_chat_log"]] == ["call_3"]
