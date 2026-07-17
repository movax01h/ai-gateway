from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node import (
    FetchNode,
)
from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    UILogEventsHumanInput,
    user_log_writer_class,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    NoneIOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.state.base import FlowEventType
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.errors.typing import InvalidRequestException


class TestFetchNode:
    """Test suite for FetchNode."""

    @pytest.fixture
    def fetch_node(self):
        """Create FetchNode instance for testing."""
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=user_log_writer_class(component_name="test_component"),
        )
        return FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
        )

    @pytest.fixture
    def sample_state(self):
        """Create sample FlowState for testing."""
        return {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {"target_agent": []},
            "ui_chat_log": [],
            "context": {},
        }

    @pytest.mark.asyncio
    async def test_interrupt_handling_response_event(self, fetch_node, sample_state):
        """Test successful interrupt handling with RESPONSE event."""
        mock_event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "User input response",
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify conversation history contains HumanMessage
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert message.content == "User input response"

            # Verify UI chat log is present and contains user response
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "User input response"
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_approve_event(self, fetch_node, sample_state):
        """Test that APPROVE event stores approval in context."""
        mock_event = {
            "event_type": FlowEventType.APPROVE,
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify approval is stored in context
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "approve"

            # Verify UI chat log is present but empty for approve events
            assert FlowStateKeys.UI_CHAT_LOG in result
            assert result[FlowStateKeys.UI_CHAT_LOG] == []

    @pytest.mark.asyncio
    async def test_interrupt_handling_reject_event_without_message(
        self, fetch_node, sample_state
    ):
        """Test that REJECT event sends instruction to agent and user-friendly message to UI."""
        mock_event = {
            "event_type": FlowEventType.REJECT,
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify rejection is stored in context
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "reject"

            # Verify default rejection message is added to conversation history
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert (
                message.content
                == "User rejected this action. Do not proceed and stop any tool execution in progress."
            )

            # Verify UI chat log contains the user-friendly rejection message
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "Action rejected."
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_modify_event(self, fetch_node, sample_state):
        """Test that MODIFY event with message adds HumanMessage to conversation history."""
        mock_event = {
            "event_type": FlowEventType.MODIFY,
            "message": "User requested modification with feedback",
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            result = await fetch_node.run(sample_state)

            # Verify status transition to EXECUTION
            assert result[FlowStateKeys.STATUS] == WorkflowStatusEnum.EXECUTION.value

            # Verify modify decision is stored in context for routing
            assert "context" in result
            assert "test_component" in result["context"]
            assert result["context"]["test_component"]["approval"] == "modify"

            # Verify HumanMessage is added to conversation history for MODIFY
            assert FlowStateKeys.CONVERSATION_HISTORY in result
            conversation = result[FlowStateKeys.CONVERSATION_HISTORY]
            assert "target_agent" in conversation
            assert len(conversation["target_agent"]) == 1

            message = conversation["target_agent"][0]
            assert isinstance(message, HumanMessage)
            assert message.content == "User requested modification with feedback"

            # Verify UI chat log contains user response
            assert FlowStateKeys.UI_CHAT_LOG in result
            ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
            assert len(ui_logs) == 1
            assert ui_logs[0]["content"] == "User requested modification with feedback"
            assert ui_logs[0]["message_type"] == "user"

    @pytest.mark.asyncio
    async def test_interrupt_handling_modify_event_without_message(
        self, fetch_node, sample_state
    ):
        """Test that MODIFY event without message raises InvalidRequestException."""
        mock_event = {
            "event_type": FlowEventType.MODIFY,
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            with pytest.raises(InvalidRequestException) as exc_info:
                await fetch_node.run(sample_state)
            assert "non-empty message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interrupt_handling_empty_response_message_raises(
        self, fetch_node, sample_state
    ):
        """A RESPONSE event with an empty message raises InvalidRequestException.

        This is the primary guard for the websocket-drop auto-retry scenario
        (https://gitlab.com/gitlab-org/gitlab/-/work_items/602799): when the
        client reconnects with an empty goal, ``Flow._resume_command`` forwards
        a ``RESPONSE`` event with an empty message to the graph, and ``FetchNode``
        raises ``InvalidRequestException`` so the workflow state is preserved and the
        caller receives ``INVALID_ARGUMENT``.
        """
        mock_event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "",
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            with pytest.raises(InvalidRequestException) as exc_info:
                await fetch_node.run(sample_state)
            assert "non-empty message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interrupt_handling_unknown_event(self, fetch_node, sample_state):
        """Test interrupt handling with unknown event type raises ValueError."""
        mock_event = {
            "event_type": "UNKNOWN_EVENT_TYPE",
        }

        with patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=mock_event,
        ):
            with pytest.raises(NotifiableAgentException) as exc_info:
                await fetch_node.run(sample_state)
            assert "unexpected event type was received" in exc_info.value.ui_message


class TestFetchNodeCancelledTurn:
    """Test suite for cancelled-turn context injection in FetchNode.

    After a stop-recovery rollback, ``Flow`` stores the discarded ``ui_chat_log``
    delta at ``context.inputs.cancelled_turn``. FetchNode prepends it to the
    model-facing HumanMessage between ``<cancelled-turn>`` meta tags and clears
    the location (consume-once). The UI log keeps the clean user message.
    """

    @pytest.fixture
    def cancelled_turn_key(self):
        return IOKey(
            target="context",
            subkeys=["inputs", "cancelled_turn"],
            optional=True,
        )

    @pytest.fixture
    def fetch_node(self, cancelled_turn_key):
        """FetchNode wired with the cancelled_turn key (as HumanInputComponent does)."""
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=user_log_writer_class(component_name="test_component"),
        )
        return FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
            cancelled_turn_key=cancelled_turn_key,
        )

    @pytest.fixture
    def cancelled_entries(self):
        return [
            {"message_type": "user", "content": "Create an http server in java"},
            {"message_type": "agent", "content": "I'll create the server in Java..."},
            {"content": "entry without a message type"},
        ]

    @pytest.fixture
    def state_with_cancelled_turn(self, cancelled_entries):
        return {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {"target_agent": []},
            "ui_chat_log": [],
            "context": {"inputs": {"cancelled_turn": cancelled_entries}},
        }

    @staticmethod
    def _patch_interrupt(event):
        return patch(
            "duo_workflow_service.agent_platform.v1.components.human_input.nodes.fetch_node.interrupt",
            return_value=event,
        )

    @pytest.mark.asyncio
    async def test_response_event_injects_cancelled_turn_block(
        self, fetch_node, state_with_cancelled_turn
    ):
        """RESPONSE event prepends the transcript in meta tags to the HumanMessage."""
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "actually create it in python",
        }

        with self._patch_interrupt(event):
            result = await fetch_node.run(state_with_cancelled_turn)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert isinstance(message, HumanMessage)
        assert message.content.startswith("<cancelled-turn>\n")
        assert "USER: Create an http server in java" in message.content
        assert "AGENT: I'll create the server in Java..." in message.content
        assert "UNKNOWN: entry without a message type" in message.content
        assert message.content.endswith(
            "</cancelled-turn>\nactually create it in python"
        )

    @pytest.mark.asyncio
    async def test_response_event_clears_cancelled_turn_after_injection(
        self, fetch_node, state_with_cancelled_turn
    ):
        """Consume-once: the state location is cleared alongside the injection."""
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "actually create it in python",
        }

        with self._patch_interrupt(event):
            result = await fetch_node.run(state_with_cancelled_turn)

        assert result["context"]["inputs"]["cancelled_turn"] is None

    @pytest.mark.asyncio
    async def test_response_event_ui_log_shows_clean_message(
        self, fetch_node, state_with_cancelled_turn
    ):
        """The UI chat log shows the user's message without the meta-tag block."""
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "actually create it in python",
        }

        with self._patch_interrupt(event):
            result = await fetch_node.run(state_with_cancelled_turn)

        ui_logs = result[FlowStateKeys.UI_CHAT_LOG]
        assert len(ui_logs) == 1
        assert ui_logs[0]["content"] == "actually create it in python"

    @pytest.mark.asyncio
    async def test_modify_event_injects_cancelled_turn_block(
        self, fetch_node, state_with_cancelled_turn
    ):
        """MODIFY events build a HumanMessage too, so they inject and clean up as well."""
        event = {
            "event_type": FlowEventType.MODIFY,
            "message": "change of plans",
        }

        with self._patch_interrupt(event):
            result = await fetch_node.run(state_with_cancelled_turn)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content.startswith("<cancelled-turn>\n")
        assert message.content.endswith("</cancelled-turn>\nchange of plans")
        assert result["context"]["inputs"]["cancelled_turn"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "context",
        [
            {},  # inputs path absent entirely (optional key resolves to None)
            {"inputs": {}},  # inputs present, no cancelled_turn
            {"inputs": {"cancelled_turn": []}},  # empty delta
        ],
    )
    async def test_response_event_without_cancelled_turn_is_clean(
        self, fetch_node, context
    ):
        """No cancelled-turn context -> clean message, no cleanup write."""
        state = {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {"target_agent": []},
            "ui_chat_log": [],
            "context": context,
        }
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "plain follow-up",
        }

        with self._patch_interrupt(event):
            result = await fetch_node.run(state)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content == "plain follow-up"
        assert "context" not in result or "inputs" not in result.get("context", {})

    @pytest.mark.asyncio
    async def test_entries_without_content_are_skipped(self, fetch_node):
        """Entries with empty content are dropped; all-empty deltas skip injection."""
        state = {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {"target_agent": []},
            "ui_chat_log": [],
            "context": {
                "inputs": {"cancelled_turn": [{"message_type": "user", "content": ""}]}
            },
        }
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "plain follow-up",
        }

        with self._patch_interrupt(event):
            result = await fetch_node.run(state)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content == "plain follow-up"
        # No cleanup write when injection was skipped
        assert "context" not in result or "inputs" not in result.get("context", {})

    @pytest.mark.asyncio
    async def test_reject_event_does_not_consume_cancelled_turn(
        self, fetch_node, state_with_cancelled_turn
    ):
        """REJECT builds a canned message: no injection, and the context stays available.

        Rejections cannot coexist with stop-recovery (approval pauses are not
        stable rollback boundaries), so this pins the degenerate behaviour.
        """
        event = {"event_type": FlowEventType.REJECT}

        with self._patch_interrupt(event):
            result = await fetch_node.run(state_with_cancelled_turn)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert "<cancelled-turn>" not in message.content
        assert "cancelled_turn" not in result["context"].get("inputs", {})

    @pytest.mark.asyncio
    async def test_runtime_iokey_cancelled_turn_key(self, state_with_cancelled_turn):
        """A RuntimeIOKey resolves its location from state for both read and cleanup."""
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=user_log_writer_class(component_name="test_component"),
        )
        runtime_key = RuntimeIOKey(
            alias="cancelled_turn",
            factory=lambda _state: IOKey(
                target="context",
                subkeys=["inputs", "cancelled_turn"],
                optional=True,
            ),
        )
        node = FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
            cancelled_turn_key=runtime_key,
        )
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "actually create it in python",
        }

        with self._patch_interrupt(event):
            result = await node.run(state_with_cancelled_turn)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content.startswith("<cancelled-turn>\n")
        assert result["context"]["inputs"]["cancelled_turn"] is None

    @pytest.mark.asyncio
    async def test_literal_cancelled_turn_key_is_ignored(
        self, state_with_cancelled_turn
    ):
        """A literal override resolves to its static text, not a ui_chat_log list.

        The shape check therefore skips injection entirely (and cleanup with it, so the literal key is never written
        back to).
        """
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=user_log_writer_class(component_name="test_component"),
        )
        node = FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
            cancelled_turn_key=IOKey(
                target="some literal value", literal=True, alias="cancelled_turn"
            ),
        )
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "plain follow-up",
        }

        with self._patch_interrupt(event):
            result = await node.run(state_with_cancelled_turn)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content == "plain follow-up"

    @pytest.mark.asyncio
    async def test_non_list_cancelled_turn_value_is_ignored(self):
        """An override pointing at a non-list state value fails the shape check."""
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=user_log_writer_class(component_name="test_component"),
        )
        node = FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
            cancelled_turn_key=IOKey(
                target="context", subkeys=["goal"], alias="cancelled_turn"
            ),
        )
        state = {
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "conversation_history": {"target_agent": []},
            "ui_chat_log": [],
            "context": {"goal": "a plain string, not a ui_chat_log delta"},
        }
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "plain follow-up",
        }

        with self._patch_interrupt(event):
            result = await node.run(state)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content == "plain follow-up"

    def test_cleanup_without_backing_location_is_noop(self, fetch_node):
        """Keys without a backing state location produce no cleanup write.

        Pins the ``_cancelled_turn_cleanup`` guard for key types that cannot be
        written back to (e.g. ``NoneIOKey``); ``run`` never reaches cleanup for
        such keys because no value can be read from them in the first place.
        """
        fetch_node._cancelled_turn_key = NoneIOKey(alias="cancelled_turn")
        assert fetch_node._cancelled_turn_cleanup({}) == {}

    @pytest.mark.asyncio
    async def test_default_none_iokey_ignores_cancelled_turn_state(
        self, state_with_cancelled_turn
    ):
        """A FetchNode built without cancelled_turn_key (backwards compat) never injects."""
        ui_history = UIHistory(
            events=[UILogEventsHumanInput.ON_USER_RESPONSE],
            writer_class=user_log_writer_class(component_name="test_component"),
        )
        node = FetchNode(
            name="test_component#fetch",
            component_name="test_component",
            output=IOKey(target="context", subkeys=["test_component", "approval"]),
            conversation_history_key=IOKey(
                target="conversation_history", subkeys=["target_agent"]
            ),
            ui_history=ui_history,
            status_key=IOKey(target="status"),
        )
        event = {
            "event_type": FlowEventType.RESPONSE,
            "message": "plain follow-up",
        }

        with self._patch_interrupt(event):
            result = await node.run(state_with_cancelled_turn)

        message = result[FlowStateKeys.CONVERSATION_HISTORY]["target_agent"][0]
        assert message.content == "plain follow-up"


def _build_resume_integration_graph():
    """Compile START -> FetchNode -> END with a real in-memory checkpointer."""
    ui_history = UIHistory(
        events=[UILogEventsHumanInput.ON_USER_RESPONSE],
        writer_class=user_log_writer_class(component_name="hi"),
    )
    node = FetchNode(
        name="hi#fetch",
        component_name="hi",
        output=IOKey(target="context", subkeys=["hi", "approval"]),
        conversation_history_key=IOKey(
            target="conversation_history", subkeys=["agent"]
        ),
        ui_history=ui_history,
        status_key=IOKey(target="status"),
    )
    graph = StateGraph(FlowState)
    graph.add_node("hi#fetch", node.run)
    graph.add_edge(START, "hi#fetch")
    graph.add_edge("hi#fetch", END)
    return graph.compile(checkpointer=InMemorySaver())


def _resume_integration_initial_state():
    return {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {"agent": []},
        "ui_chat_log": [],
        "context": {},
    }


class TestFetchNodeResumeIntegration:
    """Integration-level reproduction for gitlab-org/gitlab#602799.

    These tests drive the real ``FetchNode`` through an actual LangGraph with a real
    (in-memory) checkpointer, exercising the genuine ``interrupt()`` / resume machinery
    rather than mocking it. They pin the exact behaviour the bug fix relies on.

    When the client reconnects with an empty goal, ``Flow._resume_command`` forwards a
    ``RESPONSE`` event with an empty message to the graph.  ``FetchNode`` detects the
    missing message and raises ``InvalidRequestException`` so the workflow state is preserved
    and the caller receives ``INVALID_ARGUMENT``.
    """

    @pytest.mark.asyncio
    async def test_initial_run_pauses_at_interrupt(self):
        """The first run interrupts inside FetchNode (pauses at INPUT_REQUIRED)."""
        graph = _build_resume_integration_graph()
        config = {"configurable": {"thread_id": "t-initial"}}

        result = await graph.ainvoke(_resume_integration_initial_state(), config)

        assert "__interrupt__" in result
        assert graph.get_state(config).next == ("hi#fetch",)

    @pytest.mark.asyncio
    async def test_resume_with_empty_response_message_raises_bad_request(self):
        """An empty-message RESPONSE event raises InvalidRequestException in FetchNode.

        This is the event ``Flow._resume_command`` builds from an empty goal (the
        websocket-drop auto-retry scenario).  ``FetchNode`` raises
        ``InvalidRequestException`` so the workflow state is preserved and the caller
        receives ``INVALID_ARGUMENT``.
        """
        graph = _build_resume_integration_graph()
        config = {"configurable": {"thread_id": "t-empty-response"}}
        await graph.ainvoke(_resume_integration_initial_state(), config)

        with pytest.raises(InvalidRequestException) as exc_info:
            await graph.ainvoke(
                Command(resume={"event_type": FlowEventType.RESPONSE, "message": ""}),
                config,
            )
        assert "non-empty message" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_resume_with_none_keeps_turn_paused(self):
        """LangGraph-level behaviour: resuming with ``None`` re-pauses at the interrupt.

        This documents the underlying LangGraph mechanic: when the graph is invoked
        with ``input=None`` after a pause, it re-enters the interrupt and stays paused.
        """
        graph = _build_resume_integration_graph()
        config = {"configurable": {"thread_id": "t-none"}}
        await graph.ainvoke(_resume_integration_initial_state(), config)

        result = await graph.ainvoke(None, config)

        assert "__interrupt__" in result
        assert graph.get_state(config).next == ("hi#fetch",)

    @pytest.mark.asyncio
    async def test_resume_with_real_message_completes(self):
        """Sanity check: a non-empty RESPONSE still resumes and completes the node."""
        graph = _build_resume_integration_graph()
        config = {"configurable": {"thread_id": "t-real"}}
        await graph.ainvoke(_resume_integration_initial_state(), config)

        result = await graph.ainvoke(
            Command(resume={"event_type": FlowEventType.RESPONSE, "message": "hello"}),
            config,
        )

        assert "__interrupt__" not in result
        assert result["status"] == WorkflowStatusEnum.EXECUTION.value
        assert result["conversation_history"]["agent"][-1].content == "hello"
