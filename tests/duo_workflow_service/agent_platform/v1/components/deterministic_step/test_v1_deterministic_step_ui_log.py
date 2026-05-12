# pylint: disable=file-naming-for-tests
from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool

from duo_workflow_service.agent_platform.v1.components.deterministic_step.ui_log import (
    UILogEventsDeterministicStep,
    UILogWriterDeterministicStep,
    deterministic_step_ui_log_writer_class,
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
        return UILogWriterDeterministicStep(
            mock_callback, component_name="test_component"
        )

    @pytest.fixture(name="ui_log_writer_with_component_name")
    def ui_log_writer_with_component_name_fixture(self, mock_callback):
        """Fixture for a UILogWriterDeterministicStep instance with component_name."""
        return UILogWriterDeterministicStep(
            mock_callback, component_name="my_component"
        )

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
            name=mock_tool.name, args=tool_call_args, tool_response=tool_response
        )

    def test_log_success_with_tool_response_in_tool_info(
        self, ui_log_writer, mock_tool, mock_callback
    ):
        """Test the success method includes tool_response in ToolInfo."""
        tool_call_args = {"param1": "value1"}
        tool_response = "Tool execution result"

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            tool_response=tool_response,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["tool_info"] == ToolInfo(
            name=mock_tool.name, args=tool_call_args, tool_response=tool_response
        )

    def test_log_error(self, ui_log_writer, mock_tool, mock_callback):
        """Test the error method."""
        error = "ERROR"
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED,
            tool_response=error,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED
        assert (
            args.record["content"]
            == "An error occurred when executing the tool: Using test_tool: param1=value1, param2=value2"
        )
        assert args.record["message_type"] == MessageTypeEnum.TOOL
        assert args.record["status"] == ToolStatus.FAILURE

    def test_log_success_embeds_component_name(
        self, ui_log_writer_with_component_name, mock_tool, mock_callback
    ):
        """Test that component_name is embedded in success log entries."""
        tool_call_args = {"param1": "value1"}

        ui_log_writer_with_component_name.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] == "my_component"

    def test_log_error_embeds_component_name(
        self, ui_log_writer_with_component_name, mock_tool, mock_callback
    ):
        """Test that component_name is embedded in error log entries."""
        tool_call_args = {"param1": "value1"}

        ui_log_writer_with_component_name.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_FAILED,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] == "my_component"

    def test_log_success_subsession_id_from_kwargs(
        self, ui_log_writer_with_component_name, mock_tool, mock_callback
    ):
        """Test that subsession_id is taken from kwargs, not stored at construction."""
        tool_call_args = {"param1": "value1"}

        ui_log_writer_with_component_name.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS,
            subsession_id="sub-123",
        )

        args = mock_callback.call_args[0][0]
        assert args.record["subsession_id"] == "sub-123"

    def test_log_success_subsession_id_none_by_default(
        self, ui_log_writer_with_component_name, mock_tool, mock_callback
    ):
        """Test that subsession_id is None when not passed in kwargs."""
        tool_call_args = {"param1": "value1"}

        ui_log_writer_with_component_name.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsDeterministicStep.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.record.get("subsession_id") is None


class TestDeterministicStepUILogWriterClassFactory:
    """Test suite for deterministic_step_ui_log_writer_class factory."""

    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        return Mock()

    def test_factory_returns_callable(self):
        """Test that the factory returns a callable."""
        factory = deterministic_step_ui_log_writer_class(component_name="my_component")
        assert callable(factory)

    def test_factory_creates_writer_with_component_name(self, mock_callback):
        """Test that the factory creates a writer with the correct component_name."""
        factory = deterministic_step_ui_log_writer_class(component_name="my_component")
        writer = factory(mock_callback)
        assert isinstance(writer, UILogWriterDeterministicStep)
        assert writer._component_name == "my_component"

    def test_factory_is_compatible_with_ui_history_writer_class(self, mock_callback):
        """Test that the factory result is compatible with UIHistory.writer_class."""
        factory = deterministic_step_ui_log_writer_class(component_name="my_component")
        # UIHistory.writer_class expects a callable that takes a single UILogCallback arg
        writer = factory(mock_callback)
        assert isinstance(writer, UILogWriterDeterministicStep)
