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
from duo_workflow_service.agent_platform.v1.state import FlowState, FlowStateKeys, IOKey
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
