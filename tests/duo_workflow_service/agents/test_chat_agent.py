from datetime import datetime, timezone
from unittest.mock import ANY, Mock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.prompt_values import ChatPromptValue

from duo_workflow_service.agents.chat_agent import ChatAgent, ChatAgentPromptTemplate
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.entities.state import (
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
)
from lib.feature_flags import current_feature_flag_context
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum


@pytest.fixture
def mock_datetime(mock_now: datetime):
    with patch("duo_workflow_service.agents.chat_agent.datetime") as mock:
        mock.now.return_value = mock_now
        mock.timezone = timezone
        yield mock


@pytest.fixture
def prompt_name():
    return "Chat Agent"


@pytest.fixture
def chat_agent(model_factory, prompt_config):
    yield ChatAgent(model_factory=model_factory, config=prompt_config)


@pytest.fixture
def input():
    return {
        "conversation_history": {"Chat Agent": [HumanMessage(content="hi")]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "project": None,
        "approval": None,
    }


@pytest.mark.asyncio
async def test_run(chat_agent, input):
    result = await chat_agent.run(input)

    assert len(result["conversation_history"]["Chat Agent"]) == 1
    assert isinstance(result["conversation_history"]["Chat Agent"][0], AIMessage)
    assert result["conversation_history"]["Chat Agent"][0].content == "Hello there!"
    assert result["ui_chat_log"] == [
        UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content="Hello there!",
            timestamp=ANY,
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
        )
    ]
    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED


class TestChatAgentTrackTokensData:
    @pytest.fixture
    def unit_primitives(self):
        return ["duo_chat"]

    @pytest.fixture
    def usage_metadata(self):
        return UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)

    @pytest.mark.asyncio
    async def test_track_tokens_data(
        self, chat_agent, input, internal_event_client: Mock
    ):
        chat_agent.internal_event_client = internal_event_client

        await chat_agent.run(input)

        assert internal_event_client.track_event.call_count == 2
        assert internal_event_client.track_event.call_args_list[0] == call(
            "token_usage_duo_chat",
            category="ai_gateway.prompts.base",
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            model_engine="litellm",
            model_name="fake-model",
            model_provider="litellm",
            additional_properties=ANY,
        )
        assert internal_event_client.track_event.call_args_list[1] == call(
            event_name=EventEnum.TOKEN_PER_USER_PROMPT.value,
            additional_properties=InternalEventAdditionalProperties(
                label="Chat Agent",
                property=EventPropertyEnum.WORKFLOW_ID.value,
                value="undefined",
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
            ),
            category=CategoryEnum.WORKFLOW_CHAT.value,
        )


class TestChatAgentPromptTemplate:
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
        self,
        prompt_template_with_split_system,
        chat_workflow_state,
        mock_datetime,
    ):
        """Test that system_static and system_dynamic create separate system messages."""
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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

        expected_date = mock_datetime.now().strftime("%Y-%m-%d")
        expected_time = mock_datetime.now().strftime("%H:%M:%S")
        expected_timezone = mock_datetime.now().astimezone().tzname()

        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message, SystemMessage)
        assert expected_date in dynamic_system_message.content
        assert expected_time in dynamic_system_message.content
        assert expected_timezone in dynamic_system_message.content
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

    def test_system_prompts_without_project(self, prompt_template_with_split_system):
        """Test system prompts when no project is provided."""
        state_without_project = ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={"test_agent": [HumanMessage(content="Hello")]},
            ui_chat_log=[],
            last_human_input=None,
            project=None,
            approval=None,
        )

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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
        self,
        prompt_template_with_split_system,
        chat_workflow_state,
        mock_datetime,
    ):
        """Test that Jinja2 variables are properly resolved in both static and dynamic parts."""
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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

        expected_date = mock_datetime.now().strftime("%Y-%m-%d")
        expected_time = mock_datetime.now().strftime("%H:%M:%S")
        expected_timezone = mock_datetime.now().astimezone().tzname()

        assert expected_date in dynamic_content
        assert expected_time in dynamic_content
        assert expected_timezone in dynamic_content
        assert "Test Project" in dynamic_content
        assert "123" in dynamic_content
        assert "https://gitlab.com/test/project" in dynamic_content

    def test_conversation_history_processing(self, prompt_template_with_split_system):
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
            project=None,
            approval=None,
        )

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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

    def test_anthropic_cache_control_enabled(
        self,
        prompt_template_with_split_system,
        chat_workflow_state,
    ):
        current_feature_flag_context.set({"enable_anthropic_prompt_caching"})

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        result = template.invoke(chat_workflow_state, agent_name="test_agent")

        assert isinstance(result, ChatPromptValue)
        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3

        # Check that the static system message has cache_control
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)

        # Content should be a list with cache_control
        assert isinstance(static_system_message.content, list)
        assert len(static_system_message.content) == 1

        content_block = static_system_message.content[0]
        assert isinstance(content_block, dict)
        assert content_block["type"] == "text"
        # Fix: The text should contain the actual text content, not the SystemMessage object
        assert "GitLab Duo Chat" in content_block["text"]
        assert "<core_mission>" in content_block["text"]
        assert content_block["cache_control"] == {"type": "ephemeral"}

        # Dynamic system message should remain as plain text
        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message, SystemMessage)
        assert isinstance(dynamic_system_message.content, str)

    def test_anthropic_cache_control_disabled(
        self,
        prompt_template_with_split_system,
        chat_workflow_state,
    ):
        """Test that cache_control is NOT added when feature flag is disabled."""
        current_feature_flag_context.set({})

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        result = template.invoke(chat_workflow_state, agent_name="test_agent")

        messages = result.messages

        # Static system message should remain as plain text
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)
        assert isinstance(static_system_message.content, str)
        assert "GitLab Duo Chat" in static_system_message.content

    def test_cache_control_only_applied_to_static_system_message(
        self,
        prompt_template_with_split_system,
        chat_workflow_state,
        mock_datetime,
    ):
        current_feature_flag_context.set({"enable_anthropic_prompt_caching"})

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        result = template.invoke(chat_workflow_state, agent_name="test_agent")

        messages = result.messages

        # Static system message should have cache_control
        static_system_message = messages[0]
        assert isinstance(static_system_message.content, list)
        assert static_system_message.content[0]["cache_control"] == {
            "type": "ephemeral"
        }

        # Dynamic system message should NOT have cache_control
        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message.content, str)
        # Verify it's the dynamic content by checking for date
        expected_date = mock_datetime.now().strftime("%Y-%m-%d")
        assert expected_date in dynamic_system_message.content
