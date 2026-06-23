# pylint: disable=file-naming-for-tests,too-many-lines
import asyncio
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node import (
    ToolNode,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    agent_tools_ui_log_writer_class,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowStateKeys,
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.state.base import NoneIOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.security.prompt_security import SecurityException
from lib.context.orbit import orbit_tool_call_count, total_tool_call_count
from lib.internal_events.event_enum import CategoryEnum, EventEnum
from tests.duo_workflow_service.agent_platform.v1.components.agent.conftest import (
    assert_security_called_with,
)


@pytest.fixture(name="mock_prompt_security")
def mock_prompt_security_fixture():
    """Fixture for mocking apply_security_scanning."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.apply_security_scanning"
    ) as mock_security:
        mock_security.return_value = "Sanitized response"
        yield mock_security


@pytest.fixture(name="mock_logger")
def mock_logger_fixture():
    """Fixture for mocking structlog logger."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.structlog"
    ) as mock_structlog:
        mock_logger = Mock()
        mock_structlog.stdlib.get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture(name="mock_tool_monitoring")
def mock_tool_monitoring_fixture():
    """Fixture for mocking duo_workflow_metrics for tool operations."""
    with (
        patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.duo_workflow_metrics"
        ) as mock_metrics,
        patch(
            "duo_workflow_service.agent_platform.utils.tool_event_tracker.duo_workflow_metrics",
            mock_metrics,
        ),
    ):
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics.time_tool_call.return_value = mock_context_manager
        yield mock_metrics


@pytest.fixture(name="tool_node")
def tool_node_fixture(  # pylint: disable=unused-argument  # fixture-on-fixture ordering deps
    component_name,
    mock_toolset,
    flow_id,
    flow_type,
    ui_history,
    mock_internal_event_client,
    mock_tool_monitoring,
    mock_prompt_security,
    mock_logger,
):
    """Fixture for ToolNode instance."""
    tracker = ToolEventTracker(
        flow_id=flow_id,
        flow_type=flow_type,
        internal_event_client=mock_internal_event_client,
    )
    static_key = IOKey(
        target="conversation_history",
        subkeys=[component_name],
        optional=True,
    )
    conversation_history_key = RuntimeIOKey(
        alias="conversation_history", factory=lambda _: static_key
    )
    return ToolNode(
        name="test_tool_node",
        conversation_history_key=conversation_history_key,
        toolset=mock_toolset,
        ui_history=ui_history,
        tracker=tracker,
    )


class TestToolNode:
    """Test suite for ToolNode class focusing on the run method."""

    @pytest.mark.asyncio
    async def test_run_success_single_tool_call(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_tool_call,
        mock_prompt_security,
        ui_history,
        mock_ai_message_with_tool_calls,
    ):
        """Test successful run with single tool call."""
        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify result structure
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        # With replace mode, result includes full history (original AI message + tool responses)
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 2  # AI message + tool response
        assert messages[0] == mock_ai_message_with_tool_calls
        assert isinstance(messages[1], ToolMessage)
        assert messages[1].tool_call_id == mock_tool_call["id"]
        assert messages[1].content == "Sanitized response"

        # Verify tool execution was called
        mock_tool.ainvoke.assert_called_once_with(mock_tool_call["args"])

        # Verify security sanitization was called
        assert_security_called_with(
            mock_prompt_security, "Tool execution result", mock_tool.name
        )

        # Verify ui_history.log.success was called with the correct parameters
        ui_history.log.success.assert_called_once_with(
            tool=mock_tool,
            tool_call_args=mock_tool_call["args"],
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
            tool_response="Tool execution result",
            subsession_id=None,
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_success_multiple_tool_calls(
        self,
        tool_node,
        base_flow_state,
        component_name,
        mock_ai_message_with_multiple_tool_calls,
        mock_toolset,
    ):
        """Test successful run with multiple tool calls."""
        # Set up toolset to return different tools
        mock_tool_1 = Mock(spec=BaseTool)
        mock_tool_1.name = "tool_1"
        mock_tool_1.ainvoke = AsyncMock(return_value="Result 1")

        mock_tool_2 = Mock(spec=BaseTool)
        mock_tool_2.name = "tool_2"
        mock_tool_2.ainvoke = AsyncMock(return_value="Result 2")

        def mock_getitem(key):
            if key == "tool_1":
                return mock_tool_1
            return mock_tool_2

        mock_toolset.__getitem__ = Mock(side_effect=mock_getitem)

        # Set up flow state
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_with_multiple_tool_calls]
        }

        result = await tool_node.run(state)

        # Verify result structure (with replace mode: AI message + 2 tool responses)
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 3  # AI message + 2 tool responses
        assert messages[0] == mock_ai_message_with_multiple_tool_calls
        assert isinstance(messages[1], ToolMessage)
        assert isinstance(messages[2], ToolMessage)

        # Verify both tools were called
        mock_tool_1.ainvoke.assert_called_once_with({"param1": "value1"})
        mock_tool_2.ainvoke.assert_called_once_with({"param2": "value2"})

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tool_call")
    async def test_run_tool_not_found(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_toolset,
        mock_prompt_security,
        mock_ai_message_with_tool_calls,
    ):
        """Test run when tool is not found in toolset."""
        # Configure toolset to not contain the tool
        mock_toolset.__contains__ = Mock(return_value=False)

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify result structure (with replace mode: AI message + tool response)
        secutiry_harness_args = mock_prompt_security.call_args
        assert "Tool test_tool not found" in secutiry_harness_args[1]["response"]
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 2  # AI message + tool response
        assert messages[0] == mock_ai_message_with_tool_calls
        assert isinstance(messages[1], ToolMessage)
        assert messages[1].content == "Sanitized response"

    @pytest.mark.asyncio
    async def test_run_type_error_handling(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_tool_call,
        mock_prompt_security,
        mock_internal_event_client,
        ui_history,
        mock_tool_monitoring,
        flow_type,
        mock_ai_message_with_tool_calls,
    ):
        """Test run handles TypeError during tool execution."""
        # Configure tool to raise TypeError
        type_error = TypeError("Invalid argument type")
        mock_tool.ainvoke = AsyncMock(side_effect=type_error)
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {},
        }

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify error message in result
        secutiry_harness_args = mock_prompt_security.call_args
        assert secutiry_harness_args[1]["tool_name"] == mock_tool.name
        assert (
            "Tool test_tool execution failed due to wrong arguments"
            in secutiry_harness_args[1]["response"]
        )
        assert "The schema is:" in secutiry_harness_args[1]["response"]
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 2  # AI message + tool response
        assert messages[0] == mock_ai_message_with_tool_calls
        assert isinstance(messages[1], ToolMessage)
        assert messages[1].content == "Sanitized response"

        # Verify internal event tracking for failure
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify ui_history.log.error was called with the correct parameters
        ui_history.log.error.assert_called_once_with(
            tool=mock_tool,
            tool_call_args=mock_tool_call["args"],
            event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            tool_response="Invalid argument type",
            subsession_id=None,
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

        # Verify tool error metric was called
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=flow_type.value,
            tool_name=mock_tool.name,
            failure_reason=type(type_error).__name__,
        )

    @pytest.mark.asyncio
    async def test_run_validation_error_handling(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_tool_call,
        mock_prompt_security,
        mock_internal_event_client,
        ui_history,
        mock_tool_monitoring,
        flow_type,
        mock_ai_message_with_tool_calls,
    ):
        """Test run handles ValidationError during tool execution."""
        # Configure tool to raise ValidationError
        validation_error = ValidationError.from_exception_data(
            "ValidationError",
            [{"type": "missing", "loc": ("field",), "input": None}],
        )
        mock_tool.ainvoke = AsyncMock(side_effect=validation_error)

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify error message in result
        secutiry_harness_args = mock_prompt_security.call_args
        assert secutiry_harness_args[1]["tool_name"] == mock_tool.name
        assert (
            "Tool test_tool raised validation error"
            in secutiry_harness_args[1]["response"]
        )
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 2  # AI message + tool response
        assert messages[0] == mock_ai_message_with_tool_calls
        assert isinstance(messages[1], ToolMessage)
        assert messages[1].content == "Sanitized response"

        # Verify internal event tracking for failure
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify ui_history.log.error was called with the correct parameters
        ui_history.log.error.assert_called_once_with(
            tool=mock_tool,
            tool_call_args=mock_tool_call["args"],
            event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            tool_response=ANY,
            subsession_id=None,
        )

        assert ui_history.log.error.call_args.kwargs["tool_response"].startswith(
            "1 validation error"
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

        # Verify tool error metric was called
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=flow_type.value,
            tool_name=mock_tool.name,
            failure_reason=type(validation_error).__name__,
        )

    @pytest.mark.asyncio
    async def test_run_generic_exception_handling(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_tool_call,
        mock_prompt_security,
        mock_internal_event_client,
        ui_history,
        mock_tool_monitoring,
        flow_type,
        mock_ai_message_with_tool_calls,
    ):
        """Test run handles generic exceptions during tool execution."""
        # Configure tool to raise generic exception
        generic_error = Exception("Generic error")
        mock_tool.ainvoke = AsyncMock(side_effect=generic_error)

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify error message in result
        secutiry_harness_args = mock_prompt_security.call_args
        assert secutiry_harness_args[1]["tool_name"] == mock_tool.name
        assert (
            "Tool runtime exception due to Generic error"
            in secutiry_harness_args[1]["response"]
        )
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 2  # AI message + tool response
        assert messages[0] == mock_ai_message_with_tool_calls
        assert isinstance(messages[1], ToolMessage)
        assert messages[1].content == "Sanitized response"

        # Verify internal event tracking for failure
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify ui_history.log.error was called with the correct parameters
        ui_history.log.error.assert_called_once_with(
            tool=mock_tool,
            tool_call_args=mock_tool_call["args"],
            event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            tool_response="Generic error",
            subsession_id=None,
        )

        # Verify ui_history.pop_state_updates was called
        ui_history.pop_state_updates.assert_called_once()

        # Verify tool error metric was called
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=flow_type.value,
            tool_name=mock_tool.name,
            failure_reason=type(generic_error).__name__,
        )

    @pytest.mark.asyncio
    async def test_run_no_tool_calls(
        self,
        tool_node,
        base_flow_state,
        component_name,
        mock_ai_message_no_tool_calls,
    ):
        """Test run with message that has no tool calls."""
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_no_tool_calls]
        }

        result = await tool_node.run(state)

        # Verify result structure (with replace mode: AI message + no tool responses)
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 1  # Only AI message, no tool responses
        assert messages[0] == mock_ai_message_no_tool_calls

    @pytest.mark.asyncio
    async def test_run_tool_call_without_args(
        self,
        tool_node,
        base_flow_state,
        component_name,
        mock_tool,
    ):
        """Test run with tool call that has no args."""
        # Create tool call without args
        tool_call_no_args = {
            "name": "test_tool",
            "id": "test_tool_call_id",
        }

        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [tool_call_no_args]

        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_message]}

        result = await tool_node.run(state)

        # Verify tool was called with empty args
        mock_tool.ainvoke.assert_called_once_with({})

        # Verify result structure (with replace mode: AI message + tool response)
        messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(messages) == 2  # AI message + tool response
        assert messages[0] == mock_message
        assert isinstance(messages[1], ToolMessage)


class TestToolNodeSecurity:
    """Test suite for ToolNode security functionality."""

    @pytest.mark.asyncio
    async def test_run_security_exception_handling(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_logger,
        mock_ai_message_with_tool_calls,
    ):
        """Test run handles SecurityException during response sanitization."""
        # Configure apply_security_scanning to raise SecurityException
        security_error = SecurityException("Security validation failed")

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.apply_security_scanning"
        ) as mock_security:
            mock_security.side_effect = security_error

            # Exception is caught internally, error message is returned
            result = await tool_node.run(flow_state_with_tool_calls)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert (
                "Security validation failed for tool test_tool"
                in mock_logger.error.call_args[0][0]
            )

            # Verify error message is in the response (with replace mode: AI message + tool response)
            messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
            assert len(messages) == 2  # AI message + tool response
            assert messages[0] == mock_ai_message_with_tool_calls
            assert (
                "Security scan detected potentially malicious content"
                in messages[1].content
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tool_call")
    async def test_run_security_sanitization_success(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_ai_message_with_tool_calls,
    ):
        """Test run with successful security sanitization."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.apply_security_scanning"
        ) as mock_security:
            mock_security.return_value = "Sanitized safe response"

            result = await tool_node.run(flow_state_with_tool_calls)

            # Verify sanitization was called
            assert_security_called_with(
                mock_security, "Tool execution result", mock_tool.name
            )

            # Verify sanitized response in result (with replace mode: AI message + tool response)
            messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
            assert len(messages) == 2  # AI message + tool response
            assert messages[0] == mock_ai_message_with_tool_calls
            assert messages[1].content == "Sanitized safe response"


class TestToolNodeMonitoring:
    """Test suite for ToolNode monitoring functionality."""

    @pytest.mark.asyncio
    async def test_run_monitoring_success(
        self,
        tool_node,
        flow_state_with_tool_calls,
        mock_tool,
        mock_tool_monitoring,
    ):
        """Test run method monitoring for successful tool execution."""
        await tool_node.run(flow_state_with_tool_calls)

        # Verify monitoring was called
        mock_tool_monitoring.time_tool_call.assert_called_once_with(
            tool_name=mock_tool.name,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT.value,
        )

    @pytest.mark.asyncio
    async def test_run_monitoring_with_error(
        self,
        tool_node,
        flow_state_with_tool_calls,
        mock_tool,
        mock_tool_monitoring,
    ):
        """Test run method monitoring when tool execution fails."""
        # Configure tool to raise exception
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("Tool error"))

        await tool_node.run(flow_state_with_tool_calls)

        # Verify monitoring was still called despite error
        mock_tool_monitoring.time_tool_call.assert_called_once_with(
            tool_name=mock_tool.name,
            flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT.value,
        )


class TestToolNodeEventTracking:
    """Test suite for ToolNode internal event tracking."""

    @pytest.mark.asyncio
    async def test_run_tracks_success_event(
        self,
        tool_node,
        flow_state_with_tool_calls,
        mock_internal_event_client,
    ):
        """Test run method tracks success event."""
        await tool_node.run(flow_state_with_tool_calls)

        # Verify internal event tracking for success
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_SUCCESS.value
        assert call_args[1]["category"] == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT

    @pytest.mark.asyncio
    async def test_run_tracks_failure_event_with_extra_data(
        self,
        tool_node,
        flow_state_with_tool_calls,
        mock_tool,
        mock_internal_event_client,
    ):
        """Test run method tracks failure event with extra error data."""
        # Configure tool to raise exception
        error_message = "Specific tool error"
        mock_tool.ainvoke = AsyncMock(side_effect=Exception(error_message))

        await tool_node.run(flow_state_with_tool_calls)

        # Verify internal event tracking includes error details
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args

        # Check that additional_properties contains error information
        additional_props = call_args[1]["additional_properties"]
        assert hasattr(additional_props, "property")
        assert additional_props.property == mock_tool.name


class TestToolNodeComponentIdentity:
    """Test suite for ToolNode component_name and subsession_id attribution in UiChatLog entries.

    ``component_name`` is now embedded in the ``UILogWriterAgentTools`` writer at
    construction time (via ``agent_tools_ui_log_writer_class``).  These tests
    verify that the writer correctly stores and uses ``component_name``, and that
    ``subsession_id`` is still resolved at runtime from state via ``session_id_key``.
    """

    def _make_tool_node(
        self,
        component_name,
        mock_toolset,
        flow_id,
        flow_type,
        mock_internal_event_client,
        *,
        writer_component_name=None,
        session_id_key=NoneIOKey(alias="session_id"),
    ):
        """Helper to build a ToolNode with a real UIHistory using the writer factory."""
        tracker = ToolEventTracker(
            flow_id=flow_id,
            flow_type=flow_type,
            internal_event_client=mock_internal_event_client,
        )
        static_key = IOKey(
            target="conversation_history",
            subkeys=[component_name],
            optional=True,
        )
        conversation_history_key = RuntimeIOKey(
            alias="conversation_history", factory=lambda _: static_key
        )
        ui_history = UIHistory(
            events=[
                UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
                UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            ],
            writer_class=agent_tools_ui_log_writer_class(
                component_name=writer_component_name,
            ),
        )
        return (
            ToolNode(
                name="test_tool_node",
                conversation_history_key=conversation_history_key,
                toolset=mock_toolset,
                ui_history=ui_history,
                tracker=tracker,
                session_id_key=session_id_key,
            ),
            ui_history,
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_tool_monitoring", "mock_prompt_security", "mock_logger"
    )
    async def test_component_name_embedded_in_writer(
        self,
        component_name,
        mock_toolset,
        flow_id,
        flow_type,
        mock_internal_event_client,
        flow_state_with_tool_calls,
    ):
        """Test that component_name set in the writer is embedded in UiChatLog entries."""
        node, _ = self._make_tool_node(
            component_name,
            mock_toolset,
            flow_id,
            flow_type,
            mock_internal_event_client,
            writer_component_name=component_name,
        )

        result = await node.run(flow_state_with_tool_calls)

        # Verify the log entry has the correct component_name from the writer
        logs = result.get("ui_chat_log", [])
        assert len(logs) == 1
        assert logs[0]["component_name"] == component_name
        assert logs[0]["subsession_id"] is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_tool_monitoring", "mock_prompt_security", "mock_logger"
    )
    async def test_session_id_resolved_from_state_on_success(
        self,
        component_name,
        mock_toolset,
        flow_id,
        flow_type,
        mock_internal_event_client,
        flow_state_with_tool_calls,
    ):
        """Test that subsession_id is resolved from state via session_id_key and embedded in log entries."""
        # Build a session_id_key that reads context.active_subsession directly (no RuntimeIOKey needed)
        session_id_key = IOKey(
            target="context",
            subkeys=["supervisor", "active_subsession"],
            optional=True,
        )

        node, _ = self._make_tool_node(
            component_name,
            mock_toolset,
            flow_id,
            flow_type,
            mock_internal_event_client,
            writer_component_name=component_name,
            session_id_key=session_id_key,
        )

        # Inject active_subsession into state
        state = flow_state_with_tool_calls.copy()
        state["context"] = {
            **state.get("context", {}),
            "supervisor": {"active_subsession": 3},
        }

        result = await node.run(state)

        logs = result.get("ui_chat_log", [])
        assert len(logs) == 1
        assert logs[0]["component_name"] == component_name
        assert logs[0]["subsession_id"] == "3"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_tool_monitoring", "mock_prompt_security", "mock_logger"
    )
    async def test_component_name_embedded_in_error_log(
        self,
        component_name,
        mock_toolset,
        flow_id,
        flow_type,
        mock_internal_event_client,
        mock_tool,
        flow_state_with_tool_calls,
    ):
        """Test that component_name from the writer is embedded in error UiChatLog entries."""
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("Tool failed"))

        node, _ = self._make_tool_node(
            component_name,
            mock_toolset,
            flow_id,
            flow_type,
            mock_internal_event_client,
            writer_component_name=component_name,
        )

        result = await node.run(flow_state_with_tool_calls)

        logs = result.get("ui_chat_log", [])
        assert len(logs) == 1
        assert logs[0]["component_name"] == component_name
        assert logs[0]["subsession_id"] is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures(
        "mock_tool_monitoring", "mock_prompt_security", "mock_logger"
    )
    async def test_session_id_none_when_no_session_id_key(
        self,
        component_name,
        mock_toolset,
        flow_id,
        flow_type,
        mock_internal_event_client,
        flow_state_with_tool_calls,
    ):
        """Test that subsession_id is None when no session_id_key is provided (standalone mode)."""
        node, _ = self._make_tool_node(
            component_name,
            mock_toolset,
            flow_id,
            flow_type,
            mock_internal_event_client,
            # No session_id_key — standalone mode
        )

        result = await node.run(flow_state_with_tool_calls)

        logs = result.get("ui_chat_log", [])
        assert len(logs) == 1
        assert logs[0]["subsession_id"] is None


class TestToolNodeOrbitTracking:
    """Test suite for Orbit-specific event tracking in ToolNode."""

    @pytest.fixture(name="orbit_tool")
    def orbit_tool_fixture(self):
        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "orbit_query_graph"
        mock_tool.ainvoke = AsyncMock(return_value="Orbit query result")
        return mock_tool

    @pytest.fixture(name="orbit_toolset")
    def orbit_toolset_fixture(self, orbit_tool):
        mock_toolset = Mock()
        mock_toolset.__contains__ = Mock(return_value=True)
        mock_toolset.__getitem__ = Mock(return_value=orbit_tool)
        mock_toolset.get = Mock(return_value=orbit_tool)
        mock_toolset.bindable = [orbit_tool]
        return mock_toolset

    @pytest.fixture(name="orbit_tool_node")
    def orbit_tool_node_fixture(
        self,
        component_name,
        orbit_toolset,
        flow_id,
        flow_type,
        ui_history,
        mock_internal_event_client,
    ):
        tracker = ToolEventTracker(
            flow_id=flow_id,
            flow_type=flow_type,
            internal_event_client=mock_internal_event_client,
        )
        static_key = IOKey(
            target="conversation_history",
            subkeys=[component_name],
            optional=True,
        )
        conversation_history_key = RuntimeIOKey(
            alias="conversation_history", factory=lambda _: static_key
        )
        return ToolNode(
            name="test_tool_node",
            conversation_history_key=conversation_history_key,
            toolset=orbit_toolset,
            ui_history=ui_history,
            tracker=tracker,
        )

    @pytest.fixture(name="orbit_flow_state")
    def orbit_flow_state_fixture(self, base_flow_state, component_name):
        orbit_tool_call = {
            "name": "orbit_query_graph",
            "args": {"query": "MATCH (p:Project) RETURN p"},
            "id": "orbit_tool_call_id",
        }
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [orbit_tool_call]
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [mock_message]}
        return state

    @pytest.fixture(autouse=True)
    def reset_orbit_counters(self):
        orbit_tool_call_count.set(0)
        total_tool_call_count.set(0)
        yield

    @pytest.mark.asyncio
    async def test_orbit_tool_success_fires_orbit_event(
        self,
        orbit_tool_node,
        orbit_flow_state,
        mock_internal_event_client,
    ):
        """Orbit tool success fires both WORKFLOW_TOOL_SUCCESS and ORBIT_DAP_TOOL_CALLED."""
        await orbit_tool_node.run(orbit_flow_state)

        assert mock_internal_event_client.track_event.call_count == 2
        event_names = [
            call[1]["event_name"]
            for call in mock_internal_event_client.track_event.call_args_list
        ]
        assert EventEnum.WORKFLOW_TOOL_SUCCESS.value in event_names
        assert EventEnum.ORBIT_DAP_TOOL_CALLED.value in event_names

    @pytest.mark.asyncio
    async def test_orbit_tool_success_increments_counters(
        self,
        orbit_tool_node,
        orbit_flow_state,
    ):
        """Orbit tool success increments both orbit and total counters."""
        await orbit_tool_node.run(orbit_flow_state)

        assert orbit_tool_call_count.get() == 1
        assert total_tool_call_count.get() == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error_kind",
        ["generic_exception", "type_error", "validation_error"],
    )
    async def test_orbit_tool_failure_fires_orbit_failed_event(
        self,
        orbit_tool_node,
        orbit_flow_state,
        orbit_tool,
        mock_internal_event_client,
        error_kind,
    ):
        """Orbit failure fires WORKFLOW_TOOL_FAILURE + ORBIT_DAP_TOOL_FAILED for all three error branches."""
        if error_kind == "generic_exception":
            orbit_tool.ainvoke = AsyncMock(side_effect=Exception("GKG query timeout"))
        elif error_kind == "type_error":
            orbit_tool.ainvoke = AsyncMock(side_effect=TypeError("bad arg type"))
            orbit_tool.args_schema = Mock()
            orbit_tool.args_schema.model_json_schema.return_value = {
                "type": "object",
                "properties": {},
            }
        else:
            orbit_tool.ainvoke = AsyncMock(
                side_effect=ValidationError.from_exception_data(
                    "ValidationError",
                    [{"type": "missing", "loc": ["field"], "msg": "Field required"}],
                )
            )

        await orbit_tool_node.run(orbit_flow_state)

        assert mock_internal_event_client.track_event.call_count == 2
        event_names = [
            call[1]["event_name"]
            for call in mock_internal_event_client.track_event.call_args_list
        ]
        assert EventEnum.WORKFLOW_TOOL_FAILURE.value in event_names
        assert EventEnum.ORBIT_DAP_TOOL_FAILED.value in event_names

    @pytest.mark.asyncio
    async def test_orbit_tool_failure_increments_counters(
        self,
        orbit_tool_node,
        orbit_flow_state,
        orbit_tool,
    ):
        """Orbit tool failure still increments orbit and total counters."""
        orbit_tool.ainvoke = AsyncMock(side_effect=Exception("GKG query timeout"))

        await orbit_tool_node.run(orbit_flow_state)

        assert orbit_tool_call_count.get() == 1
        assert total_tool_call_count.get() == 1

    @pytest.mark.asyncio
    async def test_non_orbit_tool_does_not_fire_orbit_events(
        self,
        tool_node,
        flow_state_with_tool_calls,
        mock_internal_event_client,
    ):
        """Non-orbit tool only fires WORKFLOW_TOOL_SUCCESS, not orbit events."""
        await tool_node.run(flow_state_with_tool_calls)

        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_SUCCESS.value

    @pytest.mark.asyncio
    async def test_non_orbit_tool_increments_total_only(
        self,
        tool_node,
        flow_state_with_tool_calls,
    ):
        """Non-orbit tool increments total counter but not orbit counter."""
        await tool_node.run(flow_state_with_tool_calls)

        assert orbit_tool_call_count.get() == 0
        assert total_tool_call_count.get() == 1

    @pytest.mark.asyncio
    async def test_orbit_event_carries_required_properties(
        self,
        orbit_tool_node,
        orbit_flow_state,
        mock_internal_event_client,
        flow_id,
    ):
        """ORBIT_DAP_TOOL_CALLED carries label/property/value/client_capabilities per YAML."""
        await orbit_tool_node.run(orbit_flow_state)

        orbit_call = next(
            call
            for call in mock_internal_event_client.track_event.call_args_list
            if call[1]["event_name"] == EventEnum.ORBIT_DAP_TOOL_CALLED.value
        )
        props = orbit_call[1]["additional_properties"]
        assert props.label == "workflow_tool_call"
        assert props.property == "orbit_query_graph"
        assert props.value == flow_id
        assert "client_capabilities" in props.extra


class TestToolNodeConcurrency:
    """Test suite verifying that multiple tool calls in one step execute concurrently."""

    def _build_node_with_toolset(
        self,
        toolset,
        component_name,
        flow_id,
        flow_type,
        ui_history,
        mock_internal_event_client,
    ):
        tracker = ToolEventTracker(
            flow_id=flow_id,
            flow_type=flow_type,
            internal_event_client=mock_internal_event_client,
        )
        static_key = IOKey(
            target="conversation_history",
            subkeys=[component_name],
            optional=True,
        )
        return ToolNode(
            name="test_tool_node",
            conversation_history_key=RuntimeIOKey(
                alias="conversation_history", factory=lambda _: static_key
            ),
            toolset=toolset,
            ui_history=ui_history,
            tracker=tracker,
        )

    def _make_state(self, base_flow_state, component_name, tool_calls):
        msg = Mock(spec=AIMessage)
        msg.tool_calls = tool_calls
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: [msg]}
        return state

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tool_monitoring", "mock_logger")
    @pytest.mark.parametrize(
        "waiter_first",
        [True, False],
        ids=["waiter_listed_first", "waiter_listed_second"],
    )
    async def test_tool_calls_executed_concurrently(
        self,
        waiter_first,
        component_name,
        base_flow_state,
        flow_id,
        flow_type,
        ui_history,
        mock_internal_event_client,
    ):
        """Prove that tool coroutines run concurrently regardless of their listing order.

        The waiter tool suspends until the signaller tool sets an event. Under sequential execution the signaller never
        gets scheduled while the waiter is suspended, so the test would hang. Completing within the timeout proves that
        both coroutines run at the same time.

        Parametrized over waiter_first so we confirm concurrency whether the blocking tool appears first or second in
        tool_calls.
        """
        signalled = asyncio.Event()

        waiter = Mock(spec=BaseTool)
        waiter.name = "waiter"

        async def _waiter(_args):
            await asyncio.wait_for(signalled.wait(), timeout=2.0)
            return "waiter_result"

        waiter.ainvoke = _waiter

        signaller = Mock(spec=BaseTool)
        signaller.name = "signaller"

        async def _signaller(_args):
            signalled.set()
            return "signaller_result"

        signaller.ainvoke = _signaller

        tools = {"waiter": waiter, "signaller": signaller}
        toolset = Mock()
        toolset.__contains__ = Mock(return_value=True)
        toolset.__getitem__ = Mock(side_effect=tools.__getitem__)
        toolset.get = Mock(side_effect=tools.get)

        node = self._build_node_with_toolset(
            toolset,
            component_name,
            flow_id,
            flow_type,
            ui_history,
            mock_internal_event_client,
        )

        ordered = (
            [
                {"name": "waiter", "args": {}, "id": "w_id"},
                {"name": "signaller", "args": {}, "id": "s_id"},
            ]
            if waiter_first
            else [
                {"name": "signaller", "args": {}, "id": "s_id"},
                {"name": "waiter", "args": {}, "id": "w_id"},
            ]
        )

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.apply_security_scanning",
            side_effect=lambda response, **_: response,
        ):
            result = await node.run(
                self._make_state(base_flow_state, component_name, ordered)
            )

        tool_messages = [
            m
            for m in result["conversation_history"][component_name]
            if isinstance(m, ToolMessage)
        ]
        by_id = {m.tool_call_id: m for m in tool_messages}
        assert by_id["w_id"].content == "waiter_result"
        assert by_id["s_id"].content == "signaller_result"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tool_monitoring", "mock_logger")
    @pytest.mark.parametrize(
        "slow_listed_first",
        [True, False],
        ids=["slow_listed_first", "fast_listed_first"],
    )
    async def test_response_order_matches_tool_call_order(
        self,
        slow_listed_first,
        component_name,
        base_flow_state,
        flow_id,
        flow_type,
        ui_history,
        mock_internal_event_client,
    ):
        """ToolMessages are returned in tool_calls order even when completion order differs.

        Parametrized so we verify ordering is preserved whether the slow tool appears first or second in the input list.
        """
        fast_done = asyncio.Event()

        tool_slow = Mock(spec=BaseTool)
        tool_slow.name = "tool_slow"

        async def _slow(_args):
            await asyncio.sleep(0)  # yield so tool_fast can run first
            await fast_done.wait()
            return "slow_result"

        tool_slow.ainvoke = _slow

        tool_fast = Mock(spec=BaseTool)
        tool_fast.name = "tool_fast"

        async def _fast(_args):
            fast_done.set()
            return "fast_result"

        tool_fast.ainvoke = _fast

        tools = {"tool_slow": tool_slow, "tool_fast": tool_fast}
        toolset = Mock()
        toolset.__contains__ = Mock(return_value=True)
        toolset.__getitem__ = Mock(side_effect=tools.__getitem__)
        toolset.get = Mock(side_effect=tools.get)

        node = self._build_node_with_toolset(
            toolset,
            component_name,
            flow_id,
            flow_type,
            ui_history,
            mock_internal_event_client,
        )

        if slow_listed_first:
            tool_calls = [
                {"name": "tool_slow", "args": {}, "id": "slow_id"},
                {"name": "tool_fast", "args": {}, "id": "fast_id"},
            ]
            expected_ids = ["slow_id", "fast_id"]
            expected_contents = ["slow_result", "fast_result"]
        else:
            tool_calls = [
                {"name": "tool_fast", "args": {}, "id": "fast_id"},
                {"name": "tool_slow", "args": {}, "id": "slow_id"},
            ]
            expected_ids = ["fast_id", "slow_id"]
            expected_contents = ["fast_result", "slow_result"]

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.apply_security_scanning",
            side_effect=lambda response, **_: response,
        ):
            result = await node.run(
                self._make_state(base_flow_state, component_name, tool_calls)
            )

        tool_messages = [
            m
            for m in result["conversation_history"][component_name]
            if isinstance(m, ToolMessage)
        ]
        assert len(tool_messages) == 2
        assert [m.tool_call_id for m in tool_messages] == expected_ids
        assert [m.content for m in tool_messages] == expected_contents

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_tool_monitoring", "mock_logger")
    async def test_exception_in_one_tool_propagates(
        self,
        component_name,
        base_flow_state,
        flow_id,
        flow_type,
        ui_history,
        mock_internal_event_client,
    ):
        """An unhandled exception raised during sanitization propagates out of run().

        _execute_tool catches all tool-invocation exceptions and returns an error string, so the exceptions that can
        escape _execute_one are those raised by _sanitize_response (e.g. an unexpected error from
        apply_security_scanning that is not a SecurityException). With return_exceptions=True all tasks are allowed to
        complete before the exception is re-raised, so the other tool is not cancelled mid-flight. The exception must
        still surface to the caller so the workflow can handle the failure correctly.
        """
        tool_ok = Mock(spec=BaseTool)
        tool_ok.name = "tool_ok"

        async def _ok(_args):
            return "ok_result"

        tool_ok.ainvoke = _ok

        tool_bad = Mock(spec=BaseTool)
        tool_bad.name = "tool_bad"

        async def _bad(_args):
            return "bad_result"

        tool_bad.ainvoke = _bad

        tools = {"tool_ok": tool_ok, "tool_bad": tool_bad}
        toolset = Mock()
        toolset.__contains__ = Mock(return_value=True)
        toolset.__getitem__ = Mock(side_effect=tools.__getitem__)
        toolset.get = Mock(side_effect=tools.get)

        node = self._build_node_with_toolset(
            toolset,
            component_name,
            flow_id,
            flow_type,
            ui_history,
            mock_internal_event_client,
        )

        tool_calls = [
            {"name": "tool_ok", "args": {}, "id": "ok_id"},
            {"name": "tool_bad", "args": {}, "id": "bad_id"},
        ]

        boom = RuntimeError("sanitization exploded")

        def _scanning_side_effect(response, **_kwargs):
            if response == "bad_result":
                raise boom
            return response

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.apply_security_scanning",
            side_effect=_scanning_side_effect,
        ):
            with pytest.raises(RuntimeError, match="sanitization exploded"):
                await node.run(
                    self._make_state(base_flow_state, component_name, tool_calls)
                )
