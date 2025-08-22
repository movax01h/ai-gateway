"""Test suite for OneOff UI logging components."""

from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool

from duo_workflow_service.agent_platform.experimental.components.one_off.ui_log import (
    UILogEventsOneOff,
    UILogWriterOneOffTools,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolInfo, ToolStatus


class TestUILogEventsOneOff:
    """Test suite for UILogEventsOneOff enum."""

    def test_events_enum_values_exist(self):
        """Test that all expected events are defined in UILogEventsOneOff."""
        expected_events = [
            "ON_AGENT_FINAL_ANSWER",
            "ON_TOOL_CALL_INPUT",
            "ON_TOOL_EXECUTION_SUCCESS",
            "ON_TOOL_EXECUTION_FAILED",
        ]

        for event_name in expected_events:
            assert hasattr(UILogEventsOneOff, event_name)
            event_value = getattr(UILogEventsOneOff, event_name)
            assert event_value is not None


class TestUILogWriterOneOffTools:
    """Test suite for UILogWriterOneOffTools class."""

    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        """Fixture for a mock callback."""
        return Mock()

    @pytest.fixture(name="ui_log_writer")
    def ui_log_writer_fixture(self, mock_callback):
        """Fixture for a UILogWriterOneOffTools instance."""
        return UILogWriterOneOffTools(mock_callback)

    @pytest.fixture(name="mock_tool")
    def mock_tool_fixture(self):
        """Fixture for a mock BaseTool instance."""
        mock = Mock(spec=BaseTool)
        mock.name = "test_tool"
        return mock

    def test_events_type_property(self, ui_log_writer):
        """Test that events_type returns the correct type."""
        assert ui_log_writer.events_type == UILogEventsOneOff

    def test_log_success(self, ui_log_writer, mock_tool, mock_callback):
        """Test the success method with a tool execution."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS
        assert "Using test_tool: param1=value1, param2=value2" in args.record["content"]
        assert args.record["message_sub_type"] == mock_tool.name
        assert args.record["message_type"] == MessageTypeEnum.TOOL
        assert args.record["status"] == ToolStatus.SUCCESS
        assert args.record["tool_info"] == ToolInfo(
            name=mock_tool.name, args=tool_call_args
        )

    def test_log_success_with_message(self, ui_log_writer, mock_tool, mock_callback):
        """Test the success method with a custom message."""
        custom_message = "Tool executed successfully with custom message"
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
            message=custom_message,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["content"] == custom_message

    def test_log_success_with_tool_response(
        self, ui_log_writer, mock_tool, mock_callback
    ):
        """Test the success method with tool_response in kwargs."""
        tool_call_args = {"param1": "value1"}
        tool_response = "Tool execution result"

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
            tool_response=tool_response,
        )

        args = mock_callback.call_args[0][0]
        # Should use the formatted message with tool response
        assert args.record["content"] == "Using test_tool: param1=value1"

    def test_log_error(self, ui_log_writer, mock_tool, mock_callback):
        """Test the error method."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED
        assert "An error occurred when executing the tool:" in args.record["content"]
        assert args.record["message_sub_type"] == mock_tool.name
        assert args.record["message_type"] == MessageTypeEnum.TOOL
        assert args.record["status"] == ToolStatus.FAILURE

    def test_log_error_with_message(self, ui_log_writer, mock_tool, mock_callback):
        """Test the error method with a custom message."""
        custom_message = "Custom error message"
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED,
            message=custom_message,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["content"] == custom_message

    def test_log_tool_call_input(self, ui_log_writer, mock_tool, mock_callback):
        """Test the _log_tool_call_input method."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        # Access the private method for testing
        ui_log_writer._log_tool_call_input(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_CALL_INPUT,
        )

        # The method returns a UiChatLog, but doesn't call the callback directly
        # Let's test it by calling log.success with the appropriate event
        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_CALL_INPUT,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsOneOff.ON_TOOL_CALL_INPUT

    def test_log_tool_call_input_pending_status(self, ui_log_writer, mock_tool):
        """Test that _log_tool_call_input creates log with PENDING status."""
        tool_call_args = {"param1": "value1"}

        result = ui_log_writer._log_tool_call_input(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_CALL_INPUT,
        )

        assert result["status"] == ToolStatus.PENDING
        assert result["message_type"] == MessageTypeEnum.TOOL
        assert result["message_sub_type"] == "test_tool_input"
        assert "Calling tool 'test_tool' with arguments:" in result["content"]

    def test_log_tool_call_input_with_custom_message(self, ui_log_writer, mock_tool):
        """Test _log_tool_call_input with custom message."""
        tool_call_args = {"param1": "value1"}
        custom_message = "Custom tool call input message"

        result = ui_log_writer._log_tool_call_input(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_CALL_INPUT,
            message=custom_message,
        )

        assert result["content"] == custom_message

    def test_format_message_with_args(self, ui_log_writer, mock_tool):
        """Test _format_message static method with tool arguments."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        result = ui_log_writer._format_message(mock_tool, tool_call_args)

        assert result == "Using test_tool: param1=value1, param2=value2"

    def test_format_message_with_empty_args(self, ui_log_writer, mock_tool):
        """Test _format_message static method with empty arguments."""
        tool_call_args = {}

        result = ui_log_writer._format_message(mock_tool, tool_call_args)

        assert result == "Using test_tool: "

    def test_format_message_with_tool_response(self, ui_log_writer, mock_tool):
        """Test _format_message static method with tool response."""
        tool_call_args = {"param1": "value1"}
        tool_response = "Tool execution result"

        result = ui_log_writer._format_message(mock_tool, tool_call_args, tool_response)

        # Should still format the same way since mock tool doesn't have format_display_message
        assert result == "Using test_tool: param1=value1"

    def test_correlation_id_and_additional_context(
        self, ui_log_writer, mock_tool, mock_callback
    ):
        """Test that correlation_id and additional_context are properly handled."""
        tool_call_args = {"param1": "value1"}
        correlation_id = "test-correlation-123"
        additional_context = [{"key": "value"}]

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_SUCCESS,
            correlation_id=correlation_id,
            context_elements=additional_context,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["correlation_id"] == correlation_id
        assert args.record["additional_context"] == additional_context

    def test_error_with_additional_context(
        self, ui_log_writer, mock_tool, mock_callback
    ):
        """Test error method with additional_context."""
        tool_call_args = {"param1": "value1"}
        additional_context = [{"error_type": "validation"}]

        ui_log_writer.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsOneOff.ON_TOOL_EXECUTION_FAILED,
            context_elements=additional_context,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["additional_context"] == additional_context
