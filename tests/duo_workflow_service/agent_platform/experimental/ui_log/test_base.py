from enum import StrEnum, auto
from unittest.mock import MagicMock

import pytest

from duo_workflow_service.agent_platform.experimental.state import FlowStateKeys
from duo_workflow_service.agent_platform.experimental.ui_log.base import (
    BaseUILogEvents,
    BaseUILogWriter,
    UIHistory,
)
from duo_workflow_service.entities import UiChatLog


# Mock events class for testing
class MockUILogEvents(BaseUILogEvents):
    ON_TEST = "on_test"
    ON_OTHER = "on_other"


# Define a mock writer class for testing
class MockWriter(BaseUILogWriter[MockUILogEvents]):
    @property
    def events_type(self) -> type[MockUILogEvents]:
        return MockUILogEvents

    def _log_success(self, message: str, **_kwargs) -> UiChatLog:
        mock_record = MagicMock(spec=UiChatLog)
        mock_record.content = message
        return mock_record


class TestBaseUILogEvents:
    def test_init_subclass_valid(self):
        # Test that a valid subclass can be created
        class ValidEvents(BaseUILogEvents):
            ON_START = auto()
            ON_END = auto()

        assert ValidEvents.ON_START == "on_start"
        assert ValidEvents.ON_END == "on_end"

    def test_init_subclass_invalid_prefix(self):
        # Test that an invalid prefix raises ValueError
        with pytest.raises(ValueError, match="All enum values must start with 'on_'"):

            class _InvalidEvents(BaseUILogEvents):
                # Missing 'on_' prefix
                START = "start"  # pylint: disable=invalid-name

    def test_init_subclass_invalid_key(self):
        # Test that an invalid key raises ValueError
        with pytest.raises(ValueError, match="Enum key .* should be .*"):

            class _InvalidKeyEvents(BaseUILogEvents):
                # Key should be 'ON_START'
                Start = "on_start"  # pylint: disable=invalid-name


class TestBaseUILogWriter:
    # Define a mock writer class for testing
    class MockWriter(BaseUILogWriter[MockUILogEvents]):
        @property
        def events_type(self) -> type[MockUILogEvents]:
            return MockUILogEvents

        def _log_success(self, message: str, **_kwargs) -> UiChatLog:
            mock_record = MagicMock(spec=UiChatLog)
            mock_record.content = message
            return mock_record

        def _log_error(self, message: str, **_kwargs) -> UiChatLog:
            mock_record = MagicMock(spec=UiChatLog)
            mock_record.content = f"ERROR: {message}"
            return mock_record

        def _log_warning(self, message: str, **_kwargs) -> UiChatLog:
            mock_record = MagicMock(spec=UiChatLog)
            mock_record.content = f"WARNING: {message}"
            return mock_record

    def test_log_callback_called(self):
        # Test that the log callback is called
        mock_callback = MagicMock()

        # Create a writer instance
        writer = self.MockWriter(mock_callback)

        # Log a message
        writer.success("Test message", event=MockUILogEvents.ON_TEST)

        # Verify callback was called
        mock_callback.assert_called_once()
        args = mock_callback.call_args[0][0]
        assert args.event == MockUILogEvents.ON_TEST
        assert args.record.content == "Test message"

    def test_missing_event(self):
        # Test that missing event raises ValueError
        writer = self.MockWriter(MagicMock())

        with pytest.raises(
            ValueError, match="Missing required keyword argument: 'event'"
        ):
            writer.success("Test message")

    def test_invalid_event_type(self):
        # Test that invalid event type raises TypeError
        mock_callback = MagicMock()
        writer = self.MockWriter(mock_callback)

        class OtherEvents(StrEnum):
            OTHER = "other"

        with pytest.raises(TypeError, match="Expected 'event' to be an instance of"):
            writer.success("Test message", event=OtherEvents.OTHER)

    def test_invalid_log_level(self):
        # Test that invalid log level raises AttributeError
        mock_callback = MagicMock()
        writer = self.MockWriter(mock_callback)

        with pytest.raises(AttributeError, match="has no log level method"):
            # Try to use a non-existent log level
            writer.info("Test message", event=MockUILogEvents.ON_TEST)

    def test_incomplete_writer(self):
        # Test that missing level method implementation raises AttributeError
        class IncompleteWriter(BaseUILogWriter[MockUILogEvents]):
            @property
            def events_type(self) -> type[MockUILogEvents]:
                return MockUILogEvents

            # Missing _log_success method

        mock_callback = MagicMock()
        writer = IncompleteWriter(mock_callback)

        with pytest.raises(NotImplementedError):
            writer.success("Test message", event=MockUILogEvents.ON_TEST)


class TestUIHistory:
    def test_history_creation(self):
        # Test creating a UIHistory instance
        history = UIHistory(writer_class=MockWriter, events=[MockUILogEvents.ON_TEST])

        # Verify the history has the correct writer class and events
        assert history.writer_class == MockWriter
        assert MockUILogEvents.ON_TEST in history.events

    def test_log_property(self):
        # Test the log property returns a writer instance
        history = UIHistory(writer_class=MockWriter, events=[MockUILogEvents.ON_TEST])

        # Verify log is an instance of MockWriter
        assert isinstance(history.log, MockWriter)

    def test_add_log_to_history(self):
        # Test adding a log to history
        history = UIHistory(writer_class=MockWriter, events=[MockUILogEvents.ON_TEST])

        # Use the writer to log a message
        history.log.success("Test message", event=MockUILogEvents.ON_TEST)

        # Verify the message was added to history
        # Get state (should flush logs)
        state = history.pop_state_updates()
        assert FlowStateKeys.UI_CHAT_LOG in state
        assert len(state[FlowStateKeys.UI_CHAT_LOG]) == 1
        assert state[FlowStateKeys.UI_CHAT_LOG][0].content == "Test message"

        # Get state again (should be empty since logs were flushed)
        second_state = history.pop_state_updates()
        assert len(second_state[FlowStateKeys.UI_CHAT_LOG]) == 0

    def test_invalid_event_log(self):
        # Test that logging with an event not in events list raises ValueError
        class OtherEvents(BaseUILogEvents):
            ON_OTHER = "on_other"

        with pytest.raises(
            TypeError,
            match="All items in 'events' must be instances of MockUILogEvents",
        ):
            _ = UIHistory(writer_class=MockWriter, events=[OtherEvents.ON_OTHER])
