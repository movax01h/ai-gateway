# pylint: disable=file-naming-for-tests
from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool

from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
    UILogWriterAgentTools,
    agent_tools_ui_log_writer_class,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolInfo, ToolStatus


class TestUILogWriterAgentTools:
    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        """Fixture for a mock callback."""
        return Mock()

    @pytest.fixture(name="ui_log_writer")
    def ui_log_writer_fixture(self, mock_callback):
        """Fixture for a UILogWriterAgentTools instance."""
        return UILogWriterAgentTools(mock_callback)

    @pytest.fixture(name="mock_tool")
    def mock_tool_fixture(self):
        """Fixture for a mock BaseTool instance."""
        mock = Mock(spec=BaseTool)
        mock.name = "test_tool"
        return mock

    def test_events_type_property(self, ui_log_writer):
        """Test that events_type returns the correct type."""
        assert ui_log_writer.events_type == UILogEventsAgent

    def test_log_success(self, ui_log_writer, mock_tool, mock_callback):
        """Test the success method with a simple tool."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS
        assert "Using test_tool: param1=value1, param2=value2" in args.record["content"]
        assert args.record["message_sub_type"] == mock_tool.name
        assert args.record["message_type"] == MessageTypeEnum.TOOL
        assert args.record["status"] == ToolStatus.SUCCESS
        assert args.record["tool_info"] == ToolInfo(
            name=mock_tool.name, args=tool_call_args
        )

    def test_log_success_with_tool_response(
        self, ui_log_writer, mock_tool, mock_callback
    ):
        """Test the success method includes tool_response in ToolInfo."""
        tool_call_args = {"param1": "value1"}
        tool_response = "Tool execution result"

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
            tool_response=tool_response,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["tool_info"] == ToolInfo(
            name=mock_tool.name, args=tool_call_args, tool_response=tool_response
        )

    def test_log_success_with_message(self, ui_log_writer, mock_tool, mock_callback):
        """Test the success method with a custom message."""
        custom_message = "Custom success message"
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
            message=custom_message,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["content"] == custom_message

    def test_log_error(self, ui_log_writer, mock_tool, mock_callback):
        """Test the error method."""
        tool_call_args = {"param1": "value1", "param2": "value2"}

        ui_log_writer.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsAgent.ON_TOOL_EXECUTION_FAILED
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
            event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            message=custom_message,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["content"] == custom_message

    @pytest.mark.parametrize(
        "method_name,event",
        [
            ("success", UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS),
            ("error", UILogEventsAgent.ON_TOOL_EXECUTION_FAILED),
        ],
    )
    def test_log_message_id_uses_supplied_value(
        self, ui_log_writer, mock_tool, mock_callback, method_name, event
    ):
        getattr(ui_log_writer, method_name)(
            tool=mock_tool,
            tool_call_args={"param1": "value1"},
            event=event,
            message_id="toolu_01ABC",
        )

        args = mock_callback.call_args[0][0]
        assert args.record["message_id"] == "toolu_01ABC"

    @pytest.mark.parametrize(
        "method_name,event",
        [
            ("success", UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS),
            ("error", UILogEventsAgent.ON_TOOL_EXECUTION_FAILED),
        ],
    )
    def test_log_message_id_defaults_to_uuid_when_absent(
        self, ui_log_writer, mock_tool, mock_callback, method_name, event
    ):
        getattr(ui_log_writer, method_name)(
            tool=mock_tool,
            tool_call_args={"param1": "value1"},
            event=event,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["message_id"].startswith("tool-")

    def test_log_success_component_name_from_constructor(
        self, mock_callback, mock_tool
    ):
        """Test that component_name set at construction is embedded in success log entries."""
        writer = UILogWriterAgentTools(mock_callback, component_name="developer")
        tool_call_args = {"param": "value"}

        writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
            subsession_id="2",
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] == "developer"
        assert args.record["subsession_id"] == "2"

    def test_log_success_component_name_defaults_to_none(
        self, ui_log_writer, mock_tool, mock_callback
    ):
        """Test that component_name defaults to None when not provided at construction."""
        tool_call_args = {"param": "value"}

        ui_log_writer.success(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] is None
        assert args.record["subsession_id"] is None

    def test_log_error_component_name_from_constructor(self, mock_callback, mock_tool):
        """Test that component_name set at construction is embedded in error log entries."""
        writer = UILogWriterAgentTools(mock_callback, component_name="researcher")
        tool_call_args = {"param": "value"}

        writer.error(
            tool=mock_tool,
            tool_call_args=tool_call_args,
            event=UILogEventsAgent.ON_TOOL_EXECUTION_FAILED,
            subsession_id="1",
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] == "researcher"
        assert args.record["subsession_id"] == "1"

    def test_log_reasoning(self, mock_callback):
        """Test the warning method emits an AGENT reasoning entry."""
        writer = UILogWriterAgentTools(mock_callback, component_name="developer")
        reasoning_text = (
            "Now let me look at the existing schema to understand the structure:"
        )

        writer.warning(
            reasoning_text,
            event=UILogEventsAgent.ON_AGENT_REASONING,
            subsession_id="42",
        )

        args = mock_callback.call_args[0][0]
        assert args.event == UILogEventsAgent.ON_AGENT_REASONING
        assert args.record["content"] == reasoning_text
        assert args.record["message_type"] == MessageTypeEnum.AGENT
        assert args.record["status"] == ToolStatus.SUCCESS
        assert args.record["tool_info"] is None
        assert args.record["message_sub_type"] == "reasoning"
        assert args.record["component_name"] == "developer"
        assert args.record["subsession_id"] == "42"

    def test_log_reasoning_without_component_name(self, mock_callback):
        """Test that reasoning log works without component_name."""
        writer = UILogWriterAgentTools(mock_callback)
        writer.warning(
            "Some reasoning",
            event=UILogEventsAgent.ON_AGENT_REASONING,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] is None
        assert args.record["subsession_id"] is None

    def test_log_reasoning_message_id_uses_supplied_value(self, mock_callback):
        """Test that the reasoning entry reuses the originating AIMessage id.

        This lets the client correlate/replace the streamed reasoning entry (keyed by the same id) instead of creating a
        duplicate entry.
        """
        writer = UILogWriterAgentTools(mock_callback)
        writer.warning(
            "Some reasoning",
            event=UILogEventsAgent.ON_AGENT_REASONING,
            message_id="lc_run--019f65e8-b75a-7141-ae41-de3b46fae734",
        )

        args = mock_callback.call_args[0][0]
        assert (
            args.record["message_id"] == "lc_run--019f65e8-b75a-7141-ae41-de3b46fae734"
        )

    def test_log_reasoning_message_id_defaults_to_uuid_when_absent(self, mock_callback):
        """Test that a random id is generated when no message_id is supplied."""
        writer = UILogWriterAgentTools(mock_callback)
        writer.warning(
            "Some reasoning",
            event=UILogEventsAgent.ON_AGENT_REASONING,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["message_id"].startswith("agent-")

    def test_agent_tools_ui_log_writer_class_factory(self, mock_callback, mock_tool):
        """Test that agent_tools_ui_log_writer_class returns a factory that creates UILogWriterAgentTools."""
        factory_fn = agent_tools_ui_log_writer_class(component_name="my_agent")
        writer = factory_fn(mock_callback)

        assert isinstance(writer, UILogWriterAgentTools)

        writer.success(
            tool=mock_tool,
            tool_call_args={},
            event=UILogEventsAgent.ON_TOOL_EXECUTION_SUCCESS,
        )

        args = mock_callback.call_args[0][0]
        assert args.record["component_name"] == "my_agent"
