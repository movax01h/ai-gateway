from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool

from duo_workflow_service.agent_platform.experimental.components.deterministic_step.ui_log import (
    UILogEventsDeterministicStep,
    UILogWriterDeterministicStep,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolInfo, ToolStatus


class TestUILogWriterDeterministicStep:
    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        """Fixture for a mock callback."""
        return Mock()

    @pytest.fixture(name="ui_log_writer")
    def ui_log_writer_fixture(self, mock_callback):
        """Fixture for a UILogWriterDeterministicStep instance."""
        return UILogWriterDeterministicStep(mock_callback)

    @pytest.fixture(name="mock_tool")
    def mock_tool_fixture(self):
        """Fixture for a mock BaseTool instance."""
        mock = Mock(spec=BaseTool)
        mock.name = "test_tool"
        return mock

    def test_events_type_property(self, ui_log_writer):
        """Test that events_type returns the correct type."""
        assert ui_log_writer.events_type == UILogEventsDeterministicStep

    def test_log_success(self, ui_log_writer, mock_tool, mock_callback):
        """Test the success method with a simple tool."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        tool_response = "response"

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            tool_response=tool_response,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS
        assert "Using test_tool: param1=value1, param2=value2" in args.record["content"]
        assert args.record["message_sub_type"] == mock_tool.name
        assert args.record["message_type"] == MessageTypeEnum.TOOL
        assert args.record["status"] == ToolStatus.SUCCESS
        assert args.record["tool_info"] == ToolInfo(
            name=mock_tool.name, args=tool_call_args
        )

    def test_log_error(self, ui_log_writer, mock_tool, mock_callback):
        """Test the error method."""
        error = "ERROR"

        ui_log_writer.error(
            tool_name=mock_tool.name,
            error=error,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED
        assert (
            f"Tool {mock_tool.name} execution failed: {error}" in args.record["content"]
        )
        assert args.record["message_type"] == MessageTypeEnum.TOOL
        assert args.record["status"] == ToolStatus.FAILURE
