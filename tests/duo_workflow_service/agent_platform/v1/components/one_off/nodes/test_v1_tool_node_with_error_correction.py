"""Test suite for ToolNodeWithErrorCorrection class."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.v1.components.one_off.nodes.tool_node_with_error_correction import (
    ToolNodeWithErrorCorrection,
)
from duo_workflow_service.agent_platform.v1.components.one_off.ui_log import (
    UILogWriterOneOffTools,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys, IOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.security.prompt_security import SecurityException
from lib.internal_events.event_enum import CategoryEnum, EventEnum


@pytest.fixture(name="mock_prompt_security")
def mock_prompt_security_fixture():
    """Fixture for mocking apply_security_scanning."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.one_off.nodes.tool_node_with_error_correction.apply_security_scanning"
    ) as mock_security:
        mock_security.return_value = "Sanitized response"
        yield mock_security


@pytest.fixture(name="mock_logger")
def mock_logger_fixture():
    """Fixture for mocking structlog logger."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.one_off.nodes.tool_node_with_error_correction.structlog"
    ) as mock_structlog:
        mock_logger = Mock()
        mock_structlog.stdlib.get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture(name="mock_tool_monitoring")
def mock_tool_monitoring_fixture():
    """Fixture for mocking duo_workflow_metrics for tool operations."""
    with patch(
        "duo_workflow_service.agent_platform.v1.components.one_off.nodes.tool_node_with_error_correction.duo_workflow_metrics"
    ) as mock_metrics:
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_metrics.time_tool_call.return_value = mock_context_manager
        yield mock_metrics


@pytest.fixture(name="ui_history_one_off")
def ui_history_one_off_fixture():
    """Fixture for UIHistory with OneOff-specific writer."""
    ui_history = Mock(spec=UIHistory)
    ui_history.log = Mock(spec=UILogWriterOneOffTools)
    ui_history.log.success = Mock()
    ui_history.log.error = Mock()
    ui_history.log._log_tool_call_input = Mock()
    ui_history.pop_state_updates = Mock(return_value={})
    return ui_history


@pytest.fixture(name="tool_calls_key")
def tool_calls_key_fixture():
    """Fixture for tool calls IOKey."""
    return IOKey(target="context", subkeys=["test_component", "tool_calls"])


@pytest.fixture(name="tool_responses_key")
def tool_responses_key_fixture():
    """Fixture for tool responses IOKey."""
    return IOKey(target="context", subkeys=["test_component", "tool_responses"])


@pytest.fixture(name="tool_node_with_error_correction")
def tool_node_with_error_correction_fixture(
    component_name,
    mock_toolset,
    flow_id,
    flow_type,
    ui_history_one_off,
    mock_internal_event_client,
    mock_tool_monitoring,
    mock_prompt_security,
    mock_logger,
    tool_calls_key,
    tool_responses_key,
):
    """Fixture for ToolNodeWithErrorCorrection instance."""
    return ToolNodeWithErrorCorrection(
        name="test_tool_node",
        component_name=component_name,
        toolset=mock_toolset,
        flow_id=flow_id,
        flow_type=flow_type,
        internal_event_client=mock_internal_event_client,
        ui_history=ui_history_one_off,
        max_correction_attempts=3,
        tool_calls_key=tool_calls_key,
        tool_responses_key=tool_responses_key,
    )


@pytest.fixture(name="flow_state_with_tool_calls_one_off")
def flow_state_with_tool_calls_one_off_fixture(
    component_name, mock_ai_message_with_tool_calls
):
    """Fixture for flow state with tool calls specific to OneOff."""
    return {
        "status": "execution",
        "conversation_history": {component_name: [mock_ai_message_with_tool_calls]},
        "ui_chat_log": [],
        "context": {},
    }


class TestToolNodeWithErrorCorrectionInitialization:
    """Test suite for ToolNodeWithErrorCorrection initialization."""

    def test_initialization_with_all_parameters(
        self, tool_node_with_error_correction, component_name, flow_id
    ):
        """Test initialization with all parameters."""
        assert tool_node_with_error_correction.name == "test_tool_node"
        assert tool_node_with_error_correction._component_name == component_name
        assert tool_node_with_error_correction._flow_id == flow_id
        assert tool_node_with_error_correction.max_correction_attempts == 3

    def test_initialization_with_defaults(
        self,
        component_name,
        mock_toolset,
        flow_id,
        flow_type,
        ui_history_one_off,
        mock_internal_event_client,
    ):
        tool_node = ToolNodeWithErrorCorrection(
            name="test_tool_node",
            component_name=component_name,
            toolset=mock_toolset,
            flow_id=flow_id,
            flow_type=flow_type,
            internal_event_client=mock_internal_event_client,
            ui_history=ui_history_one_off,
            # Omitting: max_correction_attempts, tool_calls_key, tool_responses_key, execution_result_key
        )

        assert tool_node.max_correction_attempts == 3
        assert tool_node.tool_calls_key is None
        assert tool_node.tool_responses_key is None
        assert tool_node.execution_result_key is None


class TestToolNodeWithErrorCorrectionRun:
    """Test suite for ToolNodeWithErrorCorrection run method."""

    @pytest.mark.asyncio
    async def test_run_success_single_tool_call(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        mock_tool_call,
        mock_tool_monitoring,
        mock_prompt_security,
        ui_history_one_off,
    ):
        """Test successful run with single tool call."""
        result = await tool_node_with_error_correction.run(
            flow_state_with_tool_calls_one_off
        )

        # Verify result structure
        assert FlowStateKeys.CONVERSATION_HISTORY in result
        assert component_name in result[FlowStateKeys.CONVERSATION_HISTORY]

        conversation_messages = result[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]
        assert len(conversation_messages) == 2  # ToolMessage + success message

        # Check ToolMessage
        tool_message = conversation_messages[0]
        assert isinstance(tool_message, ToolMessage)
        assert tool_message.tool_call_id == mock_tool_call["id"]
        assert tool_message.content == "Sanitized response"

        # Check success message
        success_message = conversation_messages[1]
        assert isinstance(success_message, HumanMessage)
        assert "completed successfully" in success_message.content

        # Verify tool execution was called
        mock_tool.ainvoke.assert_called_once_with(mock_tool_call["args"])

        # Verify security sanitization was called
        mock_prompt_security.assert_called_once()

        # Verify ui_history methods were called
        ui_history_one_off.log._log_tool_call_input.assert_called_once()
        ui_history_one_off.log.success.assert_called_once()
        ui_history_one_off.pop_state_updates.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_io_keys_storage(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        tool_calls_key,
        tool_responses_key,
    ):
        """Test run method stores tool calls and responses using IOKeys."""
        result = await tool_node_with_error_correction.run(
            flow_state_with_tool_calls_one_off
        )

        # Verify tool calls are stored using IOKey
        assert "context" in result
        assert "test_component" in result["context"]
        assert "tool_calls" in result["context"]["test_component"]

        # Verify tool responses are stored using IOKey
        assert "tool_responses" in result["context"]["test_component"]

    @pytest.mark.asyncio
    async def test_run_tool_not_found(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_toolset,
        mock_tool_call,
        mock_prompt_security,
    ):
        """Test run when tool is not found in toolset."""
        # Configure toolset to not contain the tool
        mock_toolset.__contains__ = Mock(return_value=False)

        result = await tool_node_with_error_correction.run(
            flow_state_with_tool_calls_one_off
        )

        # Verify error handling
        conversation_messages = result[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]
        tool_message = conversation_messages[0]
        assert isinstance(tool_message, ToolMessage)

        security_args = mock_prompt_security.call_args
        assert (
            f"Tool {mock_tool_call['name']} not found" in security_args[1]["response"]
        )

    @pytest.mark.asyncio
    async def test_run_with_tool_execution_error(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        mock_tool_call,
        ui_history_one_off,
        mock_prompt_security,
        mock_internal_event_client,
    ):
        """Test run with tool execution error."""
        # Configure tool to raise exception
        mock_tool.ainvoke = AsyncMock(side_effect=Exception("Tool execution failed"))

        await tool_node_with_error_correction.run(flow_state_with_tool_calls_one_off)

        # Check what was passed to apply_security_scanning (the actual error message)
        security_args = mock_prompt_security.call_args
        assert (
            "Tool runtime exception due to Tool execution failed"
            in security_args[1]["response"]
        )

        # Verify internal event tracking
        mock_internal_event_client.track_event.assert_called()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify error UI logging
        ui_history_one_off.log.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_type_error(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        mock_internal_event_client,
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

        await tool_node_with_error_correction.run(flow_state_with_tool_calls_one_off)

        # Verify error handling and internal event tracking
        mock_internal_event_client.track_event.assert_called()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

    @pytest.mark.asyncio
    async def test_run_with_validation_error(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        mock_internal_event_client,
    ):
        """Test run handles ValidationError during tool execution."""
        # Configure tool to raise ValidationError
        validation_error = ValidationError.from_exception_data(
            "ValidationError",
            [{"type": "missing", "loc": ["field"], "msg": "Field required"}],
        )
        mock_tool.ainvoke = AsyncMock(side_effect=validation_error)

        result = await tool_node_with_error_correction.run(
            flow_state_with_tool_calls_one_off
        )

        # Verify error handling
        conversation_messages = result[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]
        tool_message = conversation_messages[0]
        assert isinstance(tool_message, ToolMessage)

        # Verify internal event tracking
        mock_internal_event_client.track_event.assert_called()

    @pytest.mark.asyncio
    async def test_run_with_tool_exception(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        mock_internal_event_client,
        mock_prompt_security,
    ):
        """Test run handles ToolException during tool execution."""
        from langchain_core.tools import ToolException

        # Configure tool to raise ToolException
        tool_error = ToolException("Tool validation failed")
        mock_tool.ainvoke = AsyncMock(side_effect=tool_error)

        # Configure security to return the actual error message
        mock_prompt_security.return_value = (
            "Tool exception occurred due to Tool validation failed"
        )

        result = await tool_node_with_error_correction.run(
            flow_state_with_tool_calls_one_off
        )

        # Verify error handling and internal event tracking
        mock_internal_event_client.track_event.assert_called()
        call_args = mock_internal_event_client.track_event.call_args
        assert call_args[1]["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value

        # Verify the error message format
        conversation_messages = result[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]
        tool_message = conversation_messages[0]
        assert isinstance(tool_message, ToolMessage)
        # Should contain our specific ToolException format
        assert "tool exception occurred" in tool_message.content.lower()

    @pytest.mark.asyncio
    async def test_run_no_tool_calls(
        self,
        tool_node_with_error_correction,
        base_flow_state,
        component_name,
        mock_ai_message_no_tool_calls,
    ):
        """Test run with message that has no tool calls."""
        state = base_flow_state.copy()
        state["conversation_history"] = {
            component_name: [mock_ai_message_no_tool_calls]
        }

        result = await tool_node_with_error_correction.run(state)

        # Verify result structure with no tool messages
        conversation_messages = result[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]
        # Should have success message even with no tool calls
        assert len(conversation_messages) == 1
        assert isinstance(conversation_messages[0], HumanMessage)
        assert "completed successfully" in conversation_messages[0].content

    @pytest.mark.asyncio
    async def test_run_empty_conversation_history(
        self,
        tool_node_with_error_correction,
        base_flow_state,
        component_name,
    ):
        """Test run with empty conversation history."""
        state = base_flow_state.copy()
        state["conversation_history"] = {component_name: []}

        result = await tool_node_with_error_correction.run(state)

        # Should handle gracefully with success message
        conversation_messages = result[FlowStateKeys.CONVERSATION_HISTORY][
            component_name
        ]
        assert len(conversation_messages) == 1
        assert isinstance(conversation_messages[0], HumanMessage)


class TestToolNodeWithErrorCorrectionErrorHandling:
    """Test suite for error handling and correction logic."""

    def test_extract_errors_from_responses(self, tool_node_with_error_correction):
        """Test _extract_errors_from_responses method."""
        # Create tool messages with various content matching our specific error formats
        tool_exception_message = ToolMessage(
            content="Tool exception occurred due to invalid input", tool_call_id="1"
        )
        type_error_message = ToolMessage(
            content="Tool test_tool execution failed due to wrong arguments",
            tool_call_id="2",
        )
        validation_error_message = ToolMessage(
            content="Tool test_tool raised validation error: Field required",
            tool_call_id="3",
        )
        runtime_error_message = ToolMessage(
            content="Tool runtime exception due to connection timeout", tool_call_id="4"
        )
        not_found_message = ToolMessage(content="Tool xyz not found", tool_call_id="5")
        success_message = ToolMessage(
            content="Operation completed successfully", tool_call_id="6"
        )

        tool_responses = [
            tool_exception_message,
            type_error_message,
            validation_error_message,
            runtime_error_message,
            not_found_message,
            success_message,
        ]

        errors = tool_node_with_error_correction._extract_errors_from_responses(
            tool_responses
        )

        assert len(errors) == 5  # All error messages should be detected
        assert "Tool exception occurred due to invalid input" in errors
        assert "Tool test_tool execution failed due to wrong arguments" in errors
        assert "Tool test_tool raised validation error: Field required" in errors
        assert "Tool runtime exception due to connection timeout" in errors
        assert "Tool xyz not found" in errors
        assert "Operation completed successfully" not in errors

    def test_create_error_feedback(self, tool_node_with_error_correction):
        """Test _create_error_feedback method."""
        errors = ["Error: Invalid argument", "Tool not found"]
        tool_calls = [
            {"name": "tool1", "args": {"param": "value"}, "id": "1"},
            {"name": "tool2", "args": {"param2": "value2"}, "id": "2"},
        ]

        feedback = tool_node_with_error_correction._create_error_feedback(
            errors, tool_calls, 1
        )

        assert isinstance(feedback, HumanMessage)
        assert "Attempt 1/3" in feedback.content
        assert "2 attempts remaining" in feedback.content
        assert "tool1" in feedback.content
        assert "tool2" in feedback.content
        assert "Error: Invalid argument" in feedback.content
        assert "Tool not found" in feedback.content

    def test_create_error_feedback_final_attempt(self, tool_node_with_error_correction):
        """Test _create_error_feedback method on final attempt."""
        errors = ["Final error"]
        tool_calls = [{"name": "tool1", "args": {}, "id": "1"}]

        feedback = tool_node_with_error_correction._create_error_feedback(
            errors, tool_calls, 3
        )

        assert isinstance(feedback, HumanMessage)
        assert "Attempt 3/3" in feedback.content
        assert "0 attempts remaining" in feedback.content


class TestToolNodeWithErrorCorrectionSecurity:
    """Test suite for security functionality."""

    @pytest.mark.asyncio
    async def test_run_security_exception_handling(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        component_name,
        mock_tool,
        mock_logger,
    ):
        """Test run handles SecurityException during response sanitization."""
        # Configure apply_security_scanning to raise SecurityException
        security_error = SecurityException("Security validation failed")

        with patch(
            "duo_workflow_service.agent_platform.v1.components.one_off.nodes.tool_node_with_error_correction.apply_security_scanning"
        ) as mock_security:
            mock_security.side_effect = security_error

            # Exception is caught internally, error message is returned
            result = await tool_node_with_error_correction.run(
                flow_state_with_tool_calls_one_off
            )

            # Verify error was logged
            mock_logger.error.assert_called_once()
            assert (
                "Security validation failed for tool test_tool"
                in mock_logger.error.call_args[0][0]
            )

            # Verify error message is in the response
            tool_messages = result[FlowStateKeys.CONVERSATION_HISTORY][component_name]
            assert (
                "Security scan detected potentially malicious content"
                in tool_messages[0].content
            )

    def test_sanitize_response_success(self, tool_node_with_error_correction):
        """Test _sanitize_response method with successful sanitization."""
        with patch(
            "duo_workflow_service.agent_platform.v1.components.one_off.nodes.tool_node_with_error_correction.apply_security_scanning"
        ) as mock_security:
            mock_security.return_value = "Sanitized safe response"

            result = tool_node_with_error_correction._sanitize_response(
                response="Original response", tool_name="test_tool"
            )

            # Verify sanitization was called
            mock_security.assert_called_once()
            call_kwargs = mock_security.call_args[1]
            assert call_kwargs["response"] == "Original response"
            assert call_kwargs["tool_name"] == "test_tool"

            assert result == "Sanitized safe response"


class TestToolNodeWithErrorCorrectionMonitoring:
    """Test suite for monitoring and metrics functionality."""

    @pytest.mark.asyncio
    async def test_run_monitoring_success(
        self,
        tool_node_with_error_correction,
        flow_state_with_tool_calls_one_off,
        mock_tool,
        mock_tool_monitoring,
    ):
        """Test run method monitoring for successful tool execution."""
        await tool_node_with_error_correction.run(flow_state_with_tool_calls_one_off)

        # Verify monitoring was called
        mock_tool_monitoring.time_tool_call.assert_called_once_with(
            tool_name=mock_tool.name
        )

    def test_record_metric_for_tool_failure(
        self, tool_node_with_error_correction, mock_tool_monitoring
    ):
        """Test _record_metric method for tool failures."""
        from lib.internal_events import InternalEventAdditionalProperties

        additional_props = InternalEventAdditionalProperties(
            property="test_tool", error_type="TypeError"
        )

        tool_node_with_error_correction._record_metric(
            EventEnum.WORKFLOW_TOOL_FAILURE, additional_props
        )

        # Verify metrics were recorded
        mock_tool_monitoring.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type=tool_node_with_error_correction._flow_type.value,
            tool_name="test_tool",
            failure_reason="TypeError",
        )
