from unittest.mock import Mock

import pytest

from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
    UserLogWriter,
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
        return AgentLogWriter(mock_log_callback)

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
        )

        args = mock_log_callback.call_args[0][0]
        assert args.event == UILogEventsHumanInput.ON_USER_INPUT_PROMPT
        assert args.record["content"] == content
        assert args.record["correlation_id"] == correlation_id
        assert args.record["additional_context"] == additional_context
        assert args.record["message_type"] == MessageTypeEnum.AGENT
        assert args.record["status"] is None


class TestUserLogWriter:
    """Test suite for UserLogWriter class."""

    @pytest.fixture
    def user_log_writer(self, mock_log_callback):
        """Fixture for a UserLogWriter instance."""
        return UserLogWriter(mock_log_callback)

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
