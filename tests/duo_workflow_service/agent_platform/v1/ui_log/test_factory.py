from unittest.mock import MagicMock

import pytest

from duo_workflow_service.agent_platform.v1.ui_log.base import (
    BaseUILogEvents,
    UILogCallback,
)
from duo_workflow_service.agent_platform.v1.ui_log.factory import (
    DefaultUILogWriter,
    default_ui_log_writer_class,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus, build_tool_info


# Mock events class for testing
class MockUILogEvents(BaseUILogEvents):
    ON_TEST = "on_test"
    ON_OTHER = "on_other"


class TestDefaultUILogWriter:
    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        return MagicMock(spec=UILogCallback)

    @pytest.fixture(name="agent_writer")
    def agent_writer_fixture(self, mock_callback):
        return DefaultUILogWriter(
            log_callback=mock_callback,
            events_class=MockUILogEvents,
            ui_role_as=MessageTypeEnum.AGENT,
        )

    @pytest.fixture(name="tool_writer")
    def tool_writer_fixture(self, mock_callback):
        return DefaultUILogWriter(
            log_callback=mock_callback,
            events_class=MockUILogEvents,
            ui_role_as=MessageTypeEnum.TOOL,
            component_name="supervisor",
        )

    def test_initialization(self, agent_writer):
        assert agent_writer.events_type == MockUILogEvents

    def test_log_success_agent_role(self, agent_writer, mock_callback):
        agent_writer.success("Test message", event=MockUILogEvents.ON_TEST)

        mock_callback.assert_called_once()
        args = mock_callback.call_args[0][0]
        assert args.event == MockUILogEvents.ON_TEST
        assert args.record["content"] == "Test message"
        assert args.record["message_type"] == MessageTypeEnum.AGENT
        assert args.record["status"] == ToolStatus.SUCCESS

    def test_log_success_tool_role_uses_tool_message_type(
        self, tool_writer, mock_callback
    ):
        """When ui_role_as=TOOL, _log_success emits message_type=tool."""
        tool_writer.success("hello", event=MockUILogEvents.ON_TEST)

        mock_callback.assert_called_once()
        record = mock_callback.call_args[0][0].record
        assert record["message_type"] == MessageTypeEnum.TOOL
        assert record["content"] == "hello"
        assert record["status"] == ToolStatus.SUCCESS

    def test_log_error_tool_role_uses_tool_message_type(
        self, tool_writer, mock_callback
    ):
        """When ui_role_as=TOOL, _log_error emits message_type=tool with FAILURE status."""
        tool_writer.error("oops", event=MockUILogEvents.ON_TEST)

        mock_callback.assert_called_once()
        record = mock_callback.call_args[0][0].record
        assert record["message_type"] == MessageTypeEnum.TOOL
        assert record["content"] == "oops"
        assert record["status"] == ToolStatus.FAILURE

    def test_component_name_from_constructor(self, tool_writer, mock_callback):
        """component_name set at construction is embedded in every log entry."""
        tool_writer.success("msg", event=MockUILogEvents.ON_TEST)

        record = mock_callback.call_args[0][0].record
        assert record["component_name"] == "supervisor"

    def test_component_name_defaults_to_none(self, agent_writer, mock_callback):
        """When component_name is not provided, it defaults to None in the log entry."""
        agent_writer.success("msg", event=MockUILogEvents.ON_TEST)

        record = mock_callback.call_args[0][0].record
        assert record["component_name"] is None

    def test_log_success_passes_through_kwargs(self, tool_writer, mock_callback):
        """_log_success forwards tool_info, message_sub_type, and subsession_id from kwargs."""
        tool_info = build_tool_info("my_tool", {"arg": "val"})
        tool_writer.success(
            "msg",
            event=MockUILogEvents.ON_TEST,
            tool_info=tool_info,
            message_sub_type="delegation_returns",
            subsession_id="1",
        )

        record = mock_callback.call_args[0][0].record
        assert record["tool_info"] is not None
        assert record["tool_info"]["name"] == "my_tool"
        assert record["message_sub_type"] == "delegation_returns"
        assert record["component_name"] == "supervisor"
        assert record["subsession_id"] == "1"

    def test_log_error_passes_through_kwargs(self, tool_writer, mock_callback):
        """_log_error forwards tool_info, message_sub_type, and subsession_id from kwargs."""
        tool_info = build_tool_info("my_tool", {"arg": "val"})
        tool_writer.error(
            "err",
            event=MockUILogEvents.ON_TEST,
            tool_info=tool_info,
            message_sub_type="delegation_returns",
            subsession_id="1",
        )

        record = mock_callback.call_args[0][0].record
        assert record["tool_info"] is not None
        assert record["message_sub_type"] == "delegation_returns"
        assert record["component_name"] == "supervisor"
        assert record["subsession_id"] == "1"

    def test_subsession_id_from_kwarg(self, tool_writer, mock_callback):
        """subsession_id passed as kwarg to log methods is used in the log entry."""
        tool_writer.success("msg", event=MockUILogEvents.ON_TEST, subsession_id="42")

        record = mock_callback.call_args[0][0].record
        assert record["subsession_id"] == "42"

    def test_subsession_id_defaults_to_none(self, tool_writer, mock_callback):
        """When subsession_id is not passed as kwarg, it defaults to None in the log entry."""
        tool_writer.success("msg", event=MockUILogEvents.ON_TEST)

        record = mock_callback.call_args[0][0].record
        assert record["subsession_id"] is None

    def test_log_warning_raises_not_implemented(self, agent_writer):
        """_log_warning raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            agent_writer._log_warning()

    def test_default_ui_log_writer_class_factory(self):
        factory_fn = default_ui_log_writer_class(
            events_class=MockUILogEvents, ui_role_as="tool"
        )

        # Create a writer using the factory
        callback = MagicMock(spec=UILogCallback)
        writer = factory_fn(callback)

        # Verify the writer is properly configured
        assert isinstance(writer, DefaultUILogWriter)
        assert writer.events_type == MockUILogEvents

    def test_default_ui_log_writer_class_factory_with_component_name(self):
        """default_ui_log_writer_class forwards component_name to the writer."""
        factory_fn = default_ui_log_writer_class(
            events_class=MockUILogEvents,
            ui_role_as="tool",
            component_name="supervisor",
        )

        callback = MagicMock(spec=UILogCallback)
        writer = factory_fn(callback)

        assert isinstance(writer, DefaultUILogWriter)
        assert writer.events_type is MockUILogEvents
        # Verify component_name is embedded by calling success and checking the record
        writer.success("msg", event=MockUILogEvents.ON_TEST)
        record = callback.call_args[0][0].record
        assert record["component_name"] == "supervisor"
