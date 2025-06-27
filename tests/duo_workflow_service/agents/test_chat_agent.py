from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from ai_gateway.prompts.config import ModelConfig
from duo_workflow_service.agents.chat_agent import ChatAgentPromptTemplate
from duo_workflow_service.entities.state import ChatWorkflowState


class TestChatAgentPromptTemplate:
    @pytest.fixture
    def mock_model_config(self):
        return MagicMock(spec=ModelConfig)

    @pytest.fixture
    def prompt_template_with_split_system(self):
        """Prompt template with both system_static and system_dynamic parts."""
        return {
            "system_static": "You are GitLab Duo Chat, an AI coding assistant.\n\n<core_mission>\nYour primary role is collaborative programming.\n</core_mission>",
            "system_dynamic": "<context>\nThe current date is {{ current_date }}. The current time is {{ current_time }}. The user's timezone is {{ current_timezone }}.\n{%- if project %}\nHere is the project information for the current GitLab project the USER is working on:\n<project>\n<project_id>{{ project.id }}</project_id>\n<project_name>{{ project.name }}</project_name>\n<project_url>{{ project.web_url }}</project_url>\n</project>\n{%- endif %}\n</context>",
            "user": "{{ message.content }}",
        }

    @pytest.fixture
    def chat_workflow_state(self):
        """Sample chat workflow state for testing."""
        return ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={"test_agent": [HumanMessage(content="Hello")]},
            ui_chat_log=[],
            last_human_input=None,
            context_elements=[],
            project={
                "id": 123,
                "description": "Test project description",
                "name": "Test Project",
                "http_url_to_repo": "https://gitlab.com/test/project.git",
                "web_url": "https://gitlab.com/test/project",
            },
            approval=None,
        )

    def test_split_system_prompts_create_separate_messages(
        self, prompt_template_with_split_system, mock_model_config, chat_workflow_state
    ):
        """Test that system_static and system_dynamic create separate system messages."""
        template = ChatAgentPromptTemplate(
            prompt_template_with_split_system, mock_model_config
        )

        with patch("duo_workflow_service.agents.chat_agent.datetime") as mock_datetime:
            mock_now = datetime(2024, 12, 25, 14, 30, 0)
            mock_datetime_instance = MagicMock()
            mock_datetime_instance.strftime = mock_now.strftime
            mock_datetime_instance.astimezone.return_value.tzname.return_value = "UTC"
            mock_datetime.now.return_value = mock_datetime_instance

            result = template.invoke(chat_workflow_state, agent_name="test_agent")

        assert isinstance(result, ChatPromptValue)
        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3

        # First message should be static system message
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)
        assert "GitLab Duo Chat" in static_system_message.content
        assert "<core_mission>" in static_system_message.content

        # Second message should be dynamic system message
        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message, SystemMessage)
        assert "2024-12-25" in dynamic_system_message.content
        assert "14:30:00" in dynamic_system_message.content
        assert "UTC" in dynamic_system_message.content
        assert "Test Project" in dynamic_system_message.content
        assert "<project_id>123</project_id>" in dynamic_system_message.content
        assert (
            "<project_name>Test Project</project_name>"
            in dynamic_system_message.content
        )
        assert (
            "<project_url>https://gitlab.com/test/project</project_url>"
            in dynamic_system_message.content
        )

        # Third message should be user message
        user_message = messages[2]
        assert isinstance(user_message, HumanMessage)

    def test_system_prompts_without_project(
        self, prompt_template_with_split_system, mock_model_config
    ):
        """Test system prompts when no project is provided."""
        state_without_project = ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={"test_agent": [HumanMessage(content="Hello")]},
            ui_chat_log=[],
            last_human_input=None,
            context_elements=[],
            project=None,
            approval=None,
        )

        template = ChatAgentPromptTemplate(
            prompt_template_with_split_system, mock_model_config
        )

        with patch("duo_workflow_service.agents.chat_agent.datetime") as mock_datetime:
            mock_now = datetime(2024, 12, 25, 14, 30, 0)
            mock_datetime_instance = MagicMock()
            mock_datetime_instance.strftime = mock_now.strftime
            mock_datetime_instance.astimezone.return_value.tzname.return_value = "UTC"
            mock_datetime.now.return_value = mock_datetime_instance

            result = template.invoke(state_without_project, agent_name="test_agent")

        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3

        # Check dynamic system message (second message)
        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message, SystemMessage)

        # Should not contain project information when project is None
        assert "Test Project" not in dynamic_system_message.content
        assert "<project>" not in dynamic_system_message.content
        assert "project_id" not in dynamic_system_message.content

    def test_jinja2_variable_resolution(
        self, prompt_template_with_split_system, mock_model_config, chat_workflow_state
    ):
        """Test that Jinja2 variables are properly resolved in both static and dynamic parts."""
        template = ChatAgentPromptTemplate(
            prompt_template_with_split_system, mock_model_config
        )

        with patch("duo_workflow_service.agents.chat_agent.datetime") as mock_datetime:
            mock_now = datetime(2024, 12, 25, 14, 30, 0)
            mock_datetime_instance = MagicMock()
            mock_datetime_instance.strftime = mock_now.strftime
            mock_datetime_instance.astimezone.return_value.tzname.return_value = "UTC"
            mock_datetime.now.return_value = mock_datetime_instance

            result = template.invoke(chat_workflow_state, agent_name="test_agent")

        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3

        # Check dynamic system message (second message) for variable resolution
        dynamic_system_message = messages[1]
        dynamic_content = dynamic_system_message.content

        # Verify all Jinja2 variables were resolved
        assert "{{ current_date }}" not in dynamic_content
        assert "{{ current_time }}" not in dynamic_content
        assert "{{ current_timezone }}" not in dynamic_content
        assert "{{ project.name }}" not in dynamic_content
        assert "{{ project.id }}" not in dynamic_content
        assert "{{ project.web_url }}" not in dynamic_content
        assert "{%- if project %}" not in dynamic_content
        assert "{%- endif %}" not in dynamic_content

        # Verify actual values are present
        assert "2024-12-25" in dynamic_content
        assert "14:30:00" in dynamic_content
        assert "UTC" in dynamic_content
        assert "Test Project" in dynamic_content
        assert "123" in dynamic_content
        assert "https://gitlab.com/test/project" in dynamic_content

    def test_conversation_history_processing(
        self, prompt_template_with_split_system, mock_model_config
    ):
        """Test that conversation history is properly processed."""
        state_with_history = ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={
                "test_agent": [
                    HumanMessage(content="First message"),
                    HumanMessage(content="Second message"),
                ]
            },
            ui_chat_log=[],
            last_human_input=None,
            context_elements=[],
            project=None,
            approval=None,
        )

        template = ChatAgentPromptTemplate(
            prompt_template_with_split_system, mock_model_config
        )

        with patch("duo_workflow_service.agents.chat_agent.datetime") as mock_datetime:
            mock_now = datetime(2024, 12, 25, 14, 30, 0)
            mock_datetime_instance = MagicMock()
            mock_datetime_instance.strftime = mock_now.strftime
            mock_datetime_instance.astimezone.return_value.tzname.return_value = "UTC"
            mock_datetime.now.return_value = mock_datetime_instance

            result = template.invoke(state_with_history, agent_name="test_agent")

        messages = result.messages

        # Should have 2 system messages + 2 user messages from history
        assert len(messages) == 4
        assert isinstance(messages[0], SystemMessage)  # Static system message
        assert isinstance(messages[1], SystemMessage)  # Dynamic system message
        assert isinstance(messages[2], HumanMessage)  # First user message
        assert isinstance(messages[3], HumanMessage)  # Second user message
        assert messages[2].content == "First message"
        assert messages[3].content == "Second message"
