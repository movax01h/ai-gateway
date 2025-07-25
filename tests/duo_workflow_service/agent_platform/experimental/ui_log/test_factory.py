from unittest.mock import MagicMock

from duo_workflow_service.agent_platform.experimental.ui_log.base import (
    BaseUILogEvents,
    UILogCallback,
)
from duo_workflow_service.agent_platform.experimental.ui_log.factory import (
    DefaultUILogWriter,
    default_ui_log_writer_class,
)
from duo_workflow_service.entities import MessageTypeEnum, ToolStatus


# Mock events class for testing
class MockUILogEvents(BaseUILogEvents):
    ON_TEST = "on_test"
    ON_OTHER = "on_other"


class TestDefaultUILogWriter:
    def test_initialization(self):
        # Test initialization of DefaultUILogWriter
        callback = MagicMock(spec=UILogCallback)
        writer = DefaultUILogWriter(
            log_callback=callback,
            events_class=MockUILogEvents,
            ui_role_as=MessageTypeEnum.AGENT,
        )

        assert writer.events_type == MockUILogEvents

    def test_log_success(self):
        # Test the _log_success method
        mock_callback = MagicMock(spec=UILogCallback)
        writer = DefaultUILogWriter(
            log_callback=mock_callback,
            events_class=MockUILogEvents,
            ui_role_as=MessageTypeEnum.AGENT,
        )

        writer.success("Test message", event=MockUILogEvents.ON_TEST)

        mock_callback.assert_called_once()
        args = mock_callback.call_args[0][0]
        assert args.event == MockUILogEvents.ON_TEST
        assert args.record["content"] == "Test message"
        assert args.record["message_type"] == MessageTypeEnum.AGENT
        assert args.record["status"] == ToolStatus.SUCCESS

    def test_default_ui_log_writer_class(self):
        factory_fn = default_ui_log_writer_class(
            events_class=MockUILogEvents, ui_role_as="tool"
        )

        # Create a writer using the factory
        callback = MagicMock(spec=UILogCallback)
        writer = factory_fn(callback)

        # Verify the writer is properly configured
        assert isinstance(writer, DefaultUILogWriter)
        assert writer.events_type == MockUILogEvents
