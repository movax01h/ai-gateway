from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node import (
    ToolNode,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.security.prompt_security import SecurityException
from lib.internal_events.event_enum import CategoryEnum, EventEnum


@pytest.fixture(name="mock_prompt_security")
def mock_prompt_security_fixture():
    """Fixture for mocking PromptSecurity."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.PromptSecurity"
    ) as mock_security:
        mock_security.apply_security_to_tool_response.return_value = (
            "Sanitized response"
        )
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
    with patch(
        "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.duo_workflow_metrics"
    ) as mock_metrics:
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics.time_tool_call.return_value = mock_context_manager
        yield mock_metrics


@pytest.fixture(name="tool_node")
def tool_node_fixture(
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
    return ToolNode(
        name="test_tool_node",
        component_name=component_name,
        toolset=mock_toolset,
        flow_id=flow_id,
        flow_type=flow_type,
        internal_event_client=mock_internal_event_client,
        ui_history=ui_history,
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
        mock_tool_monitoring,
        mock_prompt_security,
        ui_history,
    ):
        """Test successful run with single tool call."""
        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify result structure
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].tool_call_id == mock_tool_call["id"]
        assert tool_messages[0].content == "Sanitized response"

        # Verify tool execution was called
        mock_tool.arun.assert_called_once_with(mock_tool_call["args"])

        # Verify security sanitization was called
        mock_prompt_security.apply_security_to_tool_response.assert_called_once_with(
            response="Tool execution result", tool_name=mock_tool.name
        )

        # Verify ui_history.log.success was called with the correct parameters
        ui_history.log.success.assert_called_once_with(
            tool=mock_tool,
            tool_call_args=mock_tool_call["args"],
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
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
        mock_tool_monitoring,
        mock_prompt_security,
    ):
        """Test successful run with multiple tool calls."""
        # Set up toolset to return different tools
        mock_tool_1 = Mock(spec=BaseTool)
        mock_tool_1.name = "tool_1"
        mock_tool_1.arun = AsyncMock(return_value="Result 1")

        mock_tool_2 = Mock(spec=BaseTool)
        mock_tool_2.name = "tool_2"
        mock_tool_2.arun = AsyncMock(return_value="Result 2")

        def mock_getitem(key):
            if key == "tool_1":
                return mock_tool_1
            elif key == "tool_2":
                return mock_tool_2

        mock_toolset.__getitem__ = Mock(side_effect=mock_getitem)

        # Set up flow state
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_with_multiple_tool_calls]
        }

        result = await tool_node.run(state)

        # Verify result structure
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 2

        # Verify both tools were called
        mock_tool_1.arun.assert_called_once_with({"param1": "value1"})
        mock_tool_2.arun.assert_called_once_with({"param2": "value2"})

    @pytest.mark.asyncio
    async def test_run_tool_not_found(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_toolset,
        mock_prompt_security,
        mock_tool_call,
    ):
        """Test run when tool is not found in toolset."""
        # Configure toolset to not contain the tool
        mock_toolset.__contains__ = Mock(return_value=False)

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify result structure
        secutiry_harness_args = (
            mock_prompt_security.apply_security_to_tool_response.call_args
        )
        assert "Tool test_tool not found" in secutiry_harness_args[1]["response"]
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].content == "Sanitized response"

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
    ):
        """Test run handles TypeError during tool execution."""
        # Configure tool to raise TypeError
        type_error = TypeError("Invalid argument type")
        mock_tool.arun = AsyncMock(side_effect=type_error)
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.model_json_schema.return_value = {
            "type": "object",
            "properties": {},
        }

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify error message in result
        secutiry_harness_args = (
            mock_prompt_security.apply_security_to_tool_response.call_args
        )
        assert secutiry_harness_args[1]["tool_name"] == mock_tool.name
        assert (
            "Tool test_tool execution failed due to wrong arguments"
            in secutiry_harness_args[1]["response"]
        )
        assert "The schema is:" in secutiry_harness_args[1]["response"]
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].content == "Sanitized response"

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
    ):
        """Test run handles ValidationError during tool execution."""
        # Configure tool to raise ValidationError
        validation_error = ValidationError.from_exception_data(
            "ValidationError",
            [{"type": "missing", "loc": ["field"], "msg": "Field required"}],
        )
        mock_tool.arun = AsyncMock(side_effect=validation_error)

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify error message in result
        secutiry_harness_args = (
            mock_prompt_security.apply_security_to_tool_response.call_args
        )
        assert secutiry_harness_args[1]["tool_name"] == mock_tool.name
        assert (
            "Tool test_tool raised validation error"
            in secutiry_harness_args[1]["response"]
        )
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].content == "Sanitized response"

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
    ):
        """Test run handles generic exceptions during tool execution."""
        # Configure tool to raise generic exception
        generic_error = Exception("Generic error")
        mock_tool.arun = AsyncMock(side_effect=generic_error)

        result = await tool_node.run(flow_state_with_tool_calls)

        # Verify error message in result
        secutiry_harness_args = (
            mock_prompt_security.apply_security_to_tool_response.call_args
        )
        assert secutiry_harness_args[1]["tool_name"] == mock_tool.name
        assert (
            "Tool runtime exception due to Generic error"
            in secutiry_harness_args[1]["response"]
        )
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)
        assert tool_messages[0].content == "Sanitized response"

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

        # Verify result structure with empty tool messages
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 0

    @pytest.mark.asyncio
    async def test_run_tool_call_without_args(
        self,
        tool_node,
        base_flow_state,
        component_name,
        mock_tool,
        mock_toolset,
        mock_tool_monitoring,
        mock_prompt_security,
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
        mock_tool.arun.assert_called_once_with({})

        # Verify result structure
        tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
        assert len(tool_messages) == 1
        assert isinstance(tool_messages[0], ToolMessage)


class TestToolNodeSecurity:
    """Test suite for ToolNode security functionality."""

    @pytest.mark.asyncio
    async def test_run_security_exception_handling(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_logger,
    ):
        """Test run handles SecurityException during response sanitization."""
        # Configure PromptSecurity to raise SecurityException
        security_error = SecurityException("Security validation failed")

        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.PromptSecurity"
        ) as mock_security:
            mock_security.apply_security_to_tool_response.side_effect = security_error

            with pytest.raises(SecurityException):
                await tool_node.run(flow_state_with_tool_calls)

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert (
                "Security validation failed for tool test_tool"
                in mock_logger.error.call_args[0][0]
            )

    @pytest.mark.asyncio
    async def test_run_security_sanitization_success(
        self,
        tool_node,
        flow_state_with_tool_calls,
        component_name,
        mock_tool,
        mock_tool_call,
    ):
        """Test run with successful security sanitization."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node.PromptSecurity"
        ) as mock_security:
            mock_security.apply_security_to_tool_response.return_value = (
                "Sanitized safe response"
            )

            result = await tool_node.run(flow_state_with_tool_calls)

            # Verify sanitization was called
            mock_security.apply_security_to_tool_response.assert_called_once_with(
                response="Tool execution result", tool_name=mock_tool.name
            )

            # Verify sanitized response in result
            tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
            assert tool_messages[0].content == "Sanitized safe response"


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
        mock_tool.arun = AsyncMock(side_effect=Exception("Tool error"))

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
        mock_tool,
        mock_internal_event_client,
        flow_id,
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
        mock_tool.arun = AsyncMock(side_effect=Exception(error_message))

        await tool_node.run(flow_state_with_tool_calls)

        # Verify internal event tracking includes error details
        mock_internal_event_client.track_event.assert_called_once()
        call_args = mock_internal_event_client.track_event.call_args

        # Check that additional_properties contains error information
        additional_props = call_args[1]["additional_properties"]
        assert hasattr(additional_props, "property")
        assert additional_props.property == mock_tool.name
