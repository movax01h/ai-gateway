from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.v1.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
    UserLogWriter,
    agent_log_writer_class,
    user_log_writer_class,
)
from duo_workflow_service.entities import MessageTypeEnum


@pytest.fixture(name="mock_log_callback")
def log_callback_fixture():
    """Fixture for a mock callback."""
    return Mock()


class TestAgentLogWriter:
    """Test suite for AgentLogWriter class."""

    @pytest.fixture
    def agent_log_writer(self, mock_log_callback):
        """Fixture for an AgentLogWriter instance."""
        return AgentLogWriter(mock_log_callback, component_name="test_component")

    def test_events_type_property(self, agent_log_writer):
        """Test that events_type property returns correct type."""
        assert agent_log_writer.events_type == UILogEventsHumanInput

    def test_log_success_agent_prompt(self, agent_log_writer, mock_log_callback):
        """Test the success method for agent prompts."""
        content = "Please provide detailed feedback:"
        correlation_id = "test-correlation-456"
        additional_context = {"step": "user_input", "component": "human_input"}

        agent_log_writer.success(
            content=content,
            correlation_id=correlation_id,
            additional_context=additional_context,
            event=UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            request_type="approval",
        )

        args = mock_log_callback.call_args[0][0]
        assert args.event == UILogEventsHumanInput.ON_USER_INPUT_PROMPT
        assert args.record["content"] == content
        assert args.record["correlation_id"] == correlation_id
        assert args.record["additional_context"] == additional_context
        assert args.record["message_type"] == MessageTypeEnum.REQUEST
        assert args.record["message_sub_type"] == "approval"
        assert args.record["status"] is None

    def test_log_success_embeds_component_name(self, mock_log_callback):
        """Test that component_name is embedded in success log entries."""
        writer = AgentLogWriter(mock_log_callback, component_name="my_component")

        writer.success(
            content="Please approve:",
            event=UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            request_type="approval",
        )

        args = mock_log_callback.call_args[0][0]
        assert args.record["component_name"] == "my_component"

    def test_log_success_subsession_id_from_kwargs(self, mock_log_callback):
        """Test that subsession_id is taken from kwargs, not stored at construction."""
        writer = AgentLogWriter(mock_log_callback, component_name="my_component")

        writer.success(
            content="Please approve:",
            event=UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            request_type="approval",
            subsession_id="sub-123",
        )

        args = mock_log_callback.call_args[0][0]
        assert args.record["subsession_id"] == "sub-123"

    def test_log_success_subsession_id_none_by_default(self, mock_log_callback):
        """Test that subsession_id is None when not passed in kwargs."""
        writer = AgentLogWriter(mock_log_callback, component_name="my_component")

        writer.success(
            content="Please approve:",
            event=UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
            request_type="approval",
        )

        args = mock_log_callback.call_args[0][0]
        assert args.record.get("subsession_id") is None


class TestUserLogWriter:
    """Test suite for UserLogWriter class."""

    @pytest.fixture
    def user_log_writer(self, mock_log_callback):
        """Fixture for a UserLogWriter instance."""
        return UserLogWriter(mock_log_callback, component_name="test_component")

    def test_events_type_property(self, user_log_writer):
        """Test that events_type property returns correct type."""
        assert user_log_writer.events_type == UILogEventsHumanInput

    def test_log_success_user_response(self, user_log_writer, mock_log_callback):
        """Test the success method for user responses."""
        content = "This is my response to the question"
        correlation_id = "test-correlation-789"
        additional_context = {"step": "user_response", "component": "human_input"}

        user_log_writer.success(
            content=content,
            correlation_id=correlation_id,
            additional_context=additional_context,
            event=UILogEventsHumanInput.ON_USER_RESPONSE,
        )

        args = mock_log_callback.call_args[0][0]
        assert args.event == UILogEventsHumanInput.ON_USER_RESPONSE
        assert args.record["content"] == content
        assert args.record["correlation_id"] == correlation_id
        assert args.record["additional_context"] == additional_context
        assert args.record["message_type"] == MessageTypeEnum.USER
        assert args.record["status"] is None

    def test_log_success_embeds_component_name(self, mock_log_callback):
        """Test that component_name is embedded in success log entries."""
        writer = UserLogWriter(mock_log_callback, component_name="my_component")

        writer.success(
            content="User response text",
            event=UILogEventsHumanInput.ON_USER_RESPONSE,
        )

        args = mock_log_callback.call_args[0][0]
        assert args.record["component_name"] == "my_component"

    def test_log_success_subsession_id_from_kwargs(self, mock_log_callback):
        """Test that subsession_id is taken from kwargs, not stored at construction."""
        writer = UserLogWriter(mock_log_callback, component_name="my_component")

        writer.success(
            content="User response text",
            event=UILogEventsHumanInput.ON_USER_RESPONSE,
            subsession_id="sub-456",
        )

        args = mock_log_callback.call_args[0][0]
        assert args.record["subsession_id"] == "sub-456"

    def test_log_success_subsession_id_none_by_default(self, mock_log_callback):
        """Test that subsession_id is None when not passed in kwargs."""
        writer = UserLogWriter(mock_log_callback, component_name="my_component")

        writer.success(
            content="User response text",
            event=UILogEventsHumanInput.ON_USER_RESPONSE,
        )

        args = mock_log_callback.call_args[0][0]
        assert args.record.get("subsession_id") is None


class TestAgentLogWriterClassFactory:
    """Test suite for agent_log_writer_class factory."""

    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        return Mock()

    def test_factory_returns_callable(self):
        """Test that the factory returns a callable."""
        factory = agent_log_writer_class(component_name="my_component")
        assert callable(factory)

    def test_factory_creates_writer_with_component_name(self, mock_callback):
        """Test that the factory creates a writer with the correct component_name."""
        factory = agent_log_writer_class(component_name="my_component")
        writer = factory(mock_callback)
        assert isinstance(writer, AgentLogWriter)
        assert writer._component_name == "my_component"

    def test_factory_is_compatible_with_ui_history_writer_class(self, mock_callback):
        """Test that the factory result is compatible with UIHistory.writer_class."""
        factory = agent_log_writer_class(component_name="my_component")
        # UIHistory.writer_class expects a callable that takes a single UILogCallback arg
        writer = factory(mock_callback)
        assert isinstance(writer, AgentLogWriter)


class TestUserLogWriterClassFactory:
    """Test suite for user_log_writer_class factory."""

    @pytest.fixture(name="mock_callback")
    def mock_callback_fixture(self):
        return Mock()

    def test_factory_returns_callable(self):
        """Test that the factory returns a callable."""
        factory = user_log_writer_class(component_name="my_component")
        assert callable(factory)

    def test_factory_creates_writer_with_component_name(self, mock_callback):
        """Test that the factory creates a writer with the correct component_name."""
        factory = user_log_writer_class(component_name="my_component")
        writer = factory(mock_callback)
        assert isinstance(writer, UserLogWriter)
        assert writer._component_name == "my_component"

    def test_factory_is_compatible_with_ui_history_writer_class(self, mock_callback):
        """Test that the factory result is compatible with UIHistory.writer_class."""
        factory = user_log_writer_class(component_name="my_component")
        # UIHistory.writer_class expects a callable that takes a single UILogCallback arg
        writer = factory(mock_callback)
        assert isinstance(writer, UserLogWriter)
