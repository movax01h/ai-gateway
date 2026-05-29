# pylint: disable=file-naming-for-tests,import-outside-toplevel
import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node import (
    AgentFinalOutput,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node import (
    FinalResponseNode,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum


def _make_conversation_history_key(component_name: str) -> RuntimeIOKey:
    """Helper to create a RuntimeIOKey for conversation history."""
    static_key = IOKey(
        target="conversation_history",
        subkeys=[component_name],
        optional=True,
    )
    return RuntimeIOKey(alias="conversation_history", factory=lambda _: static_key)


def _make_output_key(output_iokey: IOKey) -> RuntimeIOKey:
    """Helper to create a RuntimeIOKey for output."""
    return RuntimeIOKey(alias="final_answer", factory=lambda _: output_iokey)


class TestFinalResponseNode:
    """Test suite for FinalResponseNode class."""

    @pytest.mark.asyncio
    async def test_run_success_with_output(
        self,
        component_name,
        simple_output,
        flow_state_with_message,
        tool_call_id,
        final_response_content,
        ui_history,
    ):
        """Test successful run with output IOKey."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            component_name=component_name,
        )

        # Execute
        result = await node.run(flow_state_with_message)

        # Verify
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # Check that a ToolMessage was appended to existing history
        history_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(history_messages) == 2  # original AI message + ToolMessage
        assert isinstance(history_messages[-1], ToolMessage)
        assert history_messages[-1].content == ""
        assert history_messages[-1].tool_call_id == tool_call_id

        # Check that output was set in context
        assert "context" in result
        assert result["context"]["result"] == final_response_content

        # Verify ui_history.log.success was called with the correct parameters
        ui_history.log.success.assert_called_once_with(
            final_response_content,
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
            subsession_id=None,
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success_with_nested_output(
        self, component_name, nested_output, ui_history
    ):
        """Test successful run with nested output IOKey."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(nested_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            component_name=component_name,
        )

        # Create mock tool call with different content
        nested_response_content = "Nested task completed!"
        mock_tool_call = {
            "id": "test_tool_call_id",
            "name": AgentFinalOutput.tool_title,
            "args": {"final_response": nested_response_content},
        }

        # Create mock AI message with tool calls
        mock_ai_message = Mock(spec=AIMessage)
        mock_ai_message.tool_calls = [mock_tool_call]

        # Create state
        state: FlowState = {
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {component_name: [mock_ai_message]},
            "ui_chat_log": [],
            "context": {},
        }

        # Execute
        result = await node.run(state)

        # Verify nested structure was created
        assert "context" in result
        assert "workflow" in result["context"]
        assert "final" in result["context"]["workflow"]
        assert "response" in result["context"]["workflow"]["final"]
        assert (
            result["context"]["workflow"]["final"]["response"]
            == nested_response_content
        )

    @pytest.mark.asyncio
    async def test_run_last_messages_is_not_ai_message_raises_error(
        self, component_name, simple_output, base_flow_state, ui_history
    ):
        """Test run raises ValueError when last message is not AIMessage."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
        )

        # Create state with non-AIMessage as last message
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [Mock(spec=BaseMessage)]}

        # Execute and verify error
        with pytest.raises(NotifiableAgentException) as exc_info:
            await node.run(state)

        assert "is not of type AIMessage" in exc_info.value.internal_detail

    @pytest.mark.asyncio
    async def test_run_multiple_tool_calls_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        mock_ai_message_with_multiple_tools,
        ui_history,
    ):
        """Test run with multiple tool calls raises ValueError."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            component_name=component_name,
        )

        # Create state with multiple tool calls
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_with_multiple_tools]
        }

        # Execute and verify error
        with pytest.raises(NotifiableAgentException) as exc_info:
            await node.run(state)

        assert (
            "Too many tool calls found in the last message"
            in exc_info.value.internal_detail
        )

    @pytest.mark.asyncio
    async def test_run_no_final_response_tool_call_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        mock_ai_message_without_final_tool,
        ui_history,
    ):
        """Test run raises ValueError when no final response tool call is found."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            component_name=component_name,
        )

        # Create state with wrong tool call
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_without_final_tool]
        }

        # Execute and verify error
        with pytest.raises(NotifiableAgentException) as exc_info:
            await node.run(state)

        assert (
            "Final response tool call not found in the conversation history"
            in exc_info.value.internal_detail
        )

    def test_extract_structured_response_no_schema_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        mock_other_tool_call,
        ui_history,
    ):
        """Test _extract_structured_response raises NotifiableAgentException when response_schema is None."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [mock_other_tool_call]

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=None,
            component_name=component_name,
        )

        history_iokey = IOKey(
            target="conversation_history",
            subkeys=[component_name],
            optional=True,
        )
        output_iokey = simple_output

        with pytest.raises(NotifiableAgentException) as exc_info:
            node._extract_structured_response(
                mock_message,
                [mock_message],
                history_iokey,
                output_iokey,
                base_flow_state,
            )

        assert (
            "Response schema is required for structured response extraction"
            in exc_info.value.internal_detail
        )

    @pytest.mark.asyncio
    async def test_run_schema_mode_no_tool_calls_raises_error(
        self,
        component_name,
        simple_output,
        base_flow_state,
        ui_history,
    ):
        """Test run raises ValueError when schema is set but model returns no tool calls."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []
        mock_message.text = "Text-only response"

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            component_name=component_name,
        )

        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_message]}

        with pytest.raises(NotifiableAgentException) as exc_info:
            await node.run(state)

        assert "Response schema requires a tool call" in exc_info.value.internal_detail

    @pytest.mark.asyncio
    async def test_run_empty_tool_calls_uses_text_path(
        self,
        component_name,
        simple_output,
        base_flow_state,
        ui_history,
    ):
        """Test run with empty tool calls uses text-only response path."""
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = []
        mock_message.text = "Text-only response"

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
        )

        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_message]}

        result = await node.run(state)

        # Verify text-only path: output set, no conversation_history update
        assert "context" in result
        assert result["context"]["result"] == "Text-only response"
        assert FlowStateKeys.CONVERSATION_HISTORY not in result

        ui_history.log.success.assert_called_once_with(
            "Text-only response",
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
            subsession_id=None,
        )

    @pytest.mark.asyncio
    async def test_run_no_conversation_history_raises_error(
        self, component_name, simple_output, flow_state_no_history, ui_history
    ):
        """Test run raises NotifiableAgentException when no conversation history exists for component."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
        )

        # Execute and verify error
        with pytest.raises(NotifiableAgentException) as exc_info:
            await node.run(flow_state_no_history)

        assert (
            "No messages found for conversation history key"
            in exc_info.value.internal_detail
        )

    @pytest.mark.asyncio
    async def test_run_empty_conversation_history_raises_error(
        self, component_name, simple_output, flow_state_empty_history, ui_history
    ):
        """Test run raises NotifiableAgentException when conversation history is empty for component."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
        )

        # Execute and verify error
        with pytest.raises(NotifiableAgentException) as exc_info:
            await node.run(flow_state_empty_history)

        assert (
            "No messages found for conversation history key"
            in exc_info.value.internal_detail
        )

    @pytest.mark.asyncio
    async def test_run_with_multiple_messages_uses_last(
        self, component_name, simple_output, base_flow_state, ui_history
    ):
        """Test run uses the last message in conversation history."""
        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            component_name=component_name,
        )

        # Create first message (should be ignored)
        first_tool_call = {
            "id": "first_tool_id",
            "name": AgentFinalOutput.tool_title,
            "args": {"final_response": "First response"},
        }
        first_message = Mock(spec=AIMessage)
        first_message.tool_calls = [first_tool_call]

        # Create last message (should be used)
        last_tool_call = {
            "id": "last_tool_id",
            "name": AgentFinalOutput.tool_title,
            "args": {"final_response": "Last response"},
        }
        last_message = Mock(spec=AIMessage)
        last_message.tool_calls = [last_tool_call]

        # Create state with multiple messages
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [first_message, last_message]}

        # Execute
        result = await node.run(state)

        # Verify last message was used (history includes both messages + ToolMessage)
        history_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert history_messages[-1].tool_call_id == "last_tool_id"
        assert result["context"]["result"] == "Last response"


class TestResponseSchemaTracking:
    """Test suite for response schema tracking in FinalResponseNode."""

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.duo_workflow_metrics"
    )
    async def test_tracking_emits_metric_and_event(
        self,
        mock_metrics,
        component_name,
        simple_output,
        flow_state_with_message,
        ui_history,
    ):
        """Test that tracking emits metric and internal event."""
        from duo_workflow_service.tracking.response_schema_tracking_context import (
            response_schema_tracking_results,
        )
        from lib.events import GLReportingEventContext
        from lib.internal_events.event_enum import EventEnum

        flow_type = GLReportingEventContext("fix_pipeline", "fix_pipeline/v1", False)
        mock_event_client = Mock()

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            response_schema_tracking=True,
            component_name=component_name,
            flow_id="test_flow_123",
            flow_type=flow_type,
            internal_event_client=mock_event_client,
        )

        token = response_schema_tracking_results.set({})
        try:
            await node.run(flow_state_with_message)

            mock_metrics.count_response_schema_output.assert_called_once_with(
                flow_type="fix_pipeline",
                component_name=component_name,
            )

            mock_event_client.track_event.assert_called_once()
            call_kwargs = mock_event_client.track_event.call_args[1]
            assert (
                call_kwargs["event_name"]
                == EventEnum.WORKFLOW_RESPONSE_SCHEMA_OUTPUT.value
            )
            additional_props = call_kwargs["additional_properties"]
            assert additional_props.label == component_name
            assert call_kwargs["category"] == "fix_pipeline"
            json.loads(additional_props.property)

            # extra contains individual output fields when output is a dict
            # (AgentFinalOutput.to_output() returns a string, so extra is empty here)

            results = response_schema_tracking_results.get()
            assert component_name in results
        finally:
            response_schema_tracking_results.reset(token)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.duo_workflow_metrics"
    )
    async def test_tracking_disabled_does_not_emit(
        self,
        mock_metrics,
        component_name,
        simple_output,
        flow_state_with_message,
        ui_history,
    ):
        """Test that tracking is not emitted when response_schema_tracking=False."""
        from duo_workflow_service.tracking.response_schema_tracking_context import (
            response_schema_tracking_results,
        )

        mock_event_client = Mock()

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            response_schema_tracking=False,
            component_name=component_name,
            internal_event_client=mock_event_client,
        )

        token = response_schema_tracking_results.set({})
        try:
            await node.run(flow_state_with_message)

            mock_metrics.count_response_schema_output.assert_not_called()
            mock_event_client.track_event.assert_not_called()
            assert response_schema_tracking_results.get() == {}
        finally:
            response_schema_tracking_results.reset(token)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.duo_workflow_metrics"
    )
    async def test_tracking_without_event_client_still_emits_metric_and_contextvar(
        self,
        mock_metrics,
        component_name,
        simple_output,
        flow_state_with_message,
        ui_history,
    ):
        """Test that metric and ContextVar are set even without an internal event client."""
        from duo_workflow_service.tracking.response_schema_tracking_context import (
            response_schema_tracking_results,
        )

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            response_schema_tracking=True,
            component_name=component_name,
            flow_id="test_flow_123",
            flow_type=None,
            internal_event_client=None,
        )

        token = response_schema_tracking_results.set({})
        try:
            await node.run(flow_state_with_message)

            mock_metrics.count_response_schema_output.assert_called_once()
            results = response_schema_tracking_results.get()
            assert component_name in results
        finally:
            response_schema_tracking_results.reset(token)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.duo_workflow_metrics"
    )
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.log"
    )
    async def test_tracking_context_included_in_log(
        self,
        mock_log,
        _mock_metrics,
        component_name,
        simple_output,
        flow_state_with_message,
        ui_history,
    ):
        """Test that response_schema_tracking_context values are included in log output."""
        from duo_workflow_service.tracking.response_schema_tracking_context import (
            response_schema_tracking_results,
        )
        from lib.events import GLReportingEventContext

        flow_type = GLReportingEventContext("fix_pipeline", "fix_pipeline/v1", False)

        flow_state_with_message["context"]["project_id"] = "12345"
        flow_state_with_message["context"]["inputs"] = {
            "merge_request": {
                "url": "https://gitlab.com/group/project/-/merge_requests/1"
            }
        }

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            response_schema_tracking=True,
            response_schema_tracking_context={
                "project_id": "context:project_id",
                "merge_request_url": "context:inputs.merge_request.url",
            },
            component_name=component_name,
            flow_id="test_flow_123",
            flow_type=flow_type,
            internal_event_client=Mock(),
        )

        token = response_schema_tracking_results.set({})
        try:
            await node.run(flow_state_with_message)

            mock_log.info.assert_called_once()
            call_kwargs = mock_log.info.call_args[1]
            assert call_kwargs["project_id"] == "12345"
            assert (
                call_kwargs["merge_request_url"]
                == "https://gitlab.com/group/project/-/merge_requests/1"
            )
        finally:
            response_schema_tracking_results.reset(token)

    @pytest.mark.asyncio
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.duo_workflow_metrics"
    )
    @patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node.log"
    )
    async def test_tracking_context_missing_values_resolve_to_none(
        self,
        mock_log,
        _mock_metrics,
        component_name,
        simple_output,
        flow_state_with_message,
        ui_history,
    ):
        """Test that missing tracking context paths resolve to None gracefully."""
        from duo_workflow_service.tracking.response_schema_tracking_context import (
            response_schema_tracking_results,
        )
        from lib.events import GLReportingEventContext

        flow_type = GLReportingEventContext("fix_pipeline", "fix_pipeline/v1", False)

        node = FinalResponseNode(
            name="test_node",
            conversation_history_key=_make_conversation_history_key(component_name),
            output_key=_make_output_key(simple_output),
            ui_history=ui_history,
            response_schema=AgentFinalOutput,
            response_schema_tracking=True,
            response_schema_tracking_context={
                "project_id": "context:project_id",
                "merge_request_url": "context:inputs.merge_request.url",
            },
            component_name=component_name,
            flow_id="test_flow_123",
            flow_type=flow_type,
            internal_event_client=Mock(),
        )

        token = response_schema_tracking_results.set({})
        try:
            await node.run(flow_state_with_message)

            mock_log.info.assert_called_once()
            call_kwargs = mock_log.info.call_args[1]
            assert call_kwargs["project_id"] is None
            assert call_kwargs["merge_request_url"] is None
        finally:
            response_schema_tracking_results.reset(token)
