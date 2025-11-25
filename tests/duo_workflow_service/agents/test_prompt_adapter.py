from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.prompts import Prompt
from ai_gateway.prompts.config.models import ModelClassProvider
from duo_workflow_service.agents.prompt_adapter import (
    ChatAgentPromptTemplate,
    DefaultPromptAdapter,
)
from duo_workflow_service.entities.state import ChatWorkflowState, WorkflowStatusEnum
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext


@pytest.fixture(name="mock_datetime")
def mock_datetime_fixture():
    """Mock datetime for consistent testing."""
    mock_now = datetime(2023, 12, 25, 15, 30, 45, tzinfo=timezone.utc)
    with patch("duo_workflow_service.agents.prompt_adapter.datetime") as mock:
        mock.now.return_value = mock_now
        mock.timezone = timezone
        yield mock


@pytest.fixture(name="namespace")
def namespace_fixture():
    return Namespace(
        id=456,
        name="test-org",
        description="Test organization",
        web_url="https://gitlab.com/test-org",
    )


@pytest.fixture(name="sample_chat_workflow_state")
def sample_chat_workflow_state_fixture(project, namespace) -> ChatWorkflowState:
    """Sample ChatWorkflowState for testing."""
    return ChatWorkflowState(
        plan={"steps": []},
        status=WorkflowStatusEnum.EXECUTION,
        conversation_history={"test_agent": [HumanMessage(content="Hello")]},
        ui_chat_log=[],
        last_human_input=None,
        goal="Test goal",
        project=project,
        namespace=namespace,
        approval=None,
        preapproved_tools=None,
    )


@pytest.mark.parametrize(
    "prompt_template",
    [
        {
            "system_static": """You are GitLab Duo Chat, an AI coding assistant.

<gitlab_instance_info>
<gitlab_instance_type>{{ gitlab_instance_type }}</gitlab_instance_type>
<gitlab_instance_url>{{ gitlab_instance_url }}</gitlab_instance_url>
<gitlab_instance_version>{{ gitlab_instance_version }}</gitlab_instance_version>
</gitlab_instance_info>

<core_mission>
Your primary role is collaborative programming.
</core_mission>""",
            "system_dynamic": """<context>
The current date is {{ current_date }}. The user's timezone is
{{ current_timezone }}.
{%- if project %}
Here is the project information for the current GitLab project the USER is working on:
<project>
<project_id>{{ project.id }}</project_id>
<project_name>{{ project.name }}</project_name>
<project_url>{{ project.web_url }}</project_url>
</project>
{%- endif %}
{%- if namespace %}
<namespace>
<namespace_id>{{ namespace.id }}</namespace_id>
<namespace_name>{{ namespace.name }}</namespace_name>
<namespace_url>{{ namespace.web_url }}</namespace_url>
<namespace_description>{{ namespace.description }}</namespace_description>
</namespace>
{%- endif %}
</context>""",
            "user": "{{ message.content }}",
        }
    ],
)
class TestChatAgentPromptTemplate:
    def test_split_system_prompts_create_separate_messages(
        self,
        prompt_config,
        sample_chat_workflow_state: ChatWorkflowState,
        project: Project,
        mock_datetime,
    ):
        template = ChatAgentPromptTemplate(prompt_config)

        mock_gitlab_info = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0-ee",
        )

        with patch.object(
            GitLabServiceContext,
            "get_current_instance_info",
            return_value=mock_gitlab_info,
        ):
            result = template.invoke(
                sample_chat_workflow_state,
                agent_name="test_agent",
            )

        assert isinstance(result, ChatPromptValue)
        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3

        # First message should be static system message
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)
        assert "GitLab Duo Chat" in static_system_message.content
        assert "<core_mission>" in static_system_message.content
        assert (
            "<gitlab_instance_type>GitLab.com (SaaS)</gitlab_instance_type>"
            in static_system_message.content
        )
        assert (
            "<gitlab_instance_url>https://gitlab.com</gitlab_instance_url>"
            in static_system_message.content
        )
        assert (
            "<gitlab_instance_version>16.5.0-ee</gitlab_instance_version>"
            in static_system_message.content
        )

        # Second message should be dynamic system message
        expected_date = mock_datetime.now().strftime("%Y-%m-%d")
        expected_timezone = mock_datetime.now().astimezone().tzname()

        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message, SystemMessage)
        assert expected_date in dynamic_system_message.content
        assert expected_timezone in dynamic_system_message.content
        assert "<project_id>123</project_id>" in dynamic_system_message.content
        assert (
            f"<project_name>{project["name"]}</project_name>"
            in dynamic_system_message.content
        )
        assert (
            f"<project_url>{project["web_url"]}</project_url>"
            in dynamic_system_message.content
        )
        assert "<namespace_id>456</namespace_id>" in dynamic_system_message.content
        assert (
            "<namespace_name>test-org</namespace_name>"
            in dynamic_system_message.content
        )

        # Third message should be user message
        user_message = messages[2]
        assert isinstance(user_message, HumanMessage)

    def test_conversation_history_processing(self, prompt_config):
        state_with_history = ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={
                "test_agent": [
                    HumanMessage(content="First message"),
                    AIMessage(content="AI response"),
                    HumanMessage(content="Second message"),
                ]
            },
            ui_chat_log=[],
            last_human_input=None,
            project=None,
            namespace=None,
            approval=None,
        )

        template = ChatAgentPromptTemplate(prompt_config)

        with patch.object(
            GitLabServiceContext, "get_current_instance_info", return_value=None
        ):
            result = template.invoke(
                state_with_history, agent_name="test_agent", is_anthropic_model=False
            )

        messages = result.messages

        # Should have 2 system messages + 3 messages from history
        assert len(messages) == 5
        assert messages[2].content == "First message"
        assert messages[3].content == "AI response"
        assert messages[4].content == "Second message"

    def test_slash_command_parsing(self, prompt_config):
        state_with_slash_command = ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={
                "test_agent": [
                    HumanMessage(content="/explain this code"),
                ]
            },
            ui_chat_log=[],
            last_human_input=None,
            project=None,
            namespace=None,
            approval=None,
        )

        template = ChatAgentPromptTemplate(prompt_config)

        with patch.object(
            GitLabServiceContext, "get_current_instance_info", return_value=None
        ):
            with patch(
                "duo_workflow_service.agents.prompt_adapter.slash_command_parse"
            ) as mock_parse:
                mock_parse.return_value = ("explain", "this code")

                result = template.invoke(
                    state_with_slash_command,
                    agent_name="test_agent",
                    is_anthropic_model=False,
                )

        mock_parse.assert_called_once_with("/explain this code")
        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3


class TestDefaultPromptAdapter:
    @pytest.fixture(name="mock_prompt")
    def mock_prompt_fixture(self):
        prompt = Mock(spec=Prompt)
        prompt.name = "test_agent"
        prompt.model_provider = ModelClassProvider.ANTHROPIC
        prompt.model = AsyncMock()
        prompt.prompt_tpl = Mock()
        return prompt

    @pytest.mark.asyncio
    async def test_get_response(self, mock_prompt, sample_chat_workflow_state):
        adapter = DefaultPromptAdapter(mock_prompt)

        expected_response = AIMessage(content="Test response")
        mock_prompt.ainvoke.return_value = expected_response

        result = await adapter.get_response(sample_chat_workflow_state)

        assert result == expected_response
        mock_prompt.ainvoke.assert_called_once_with(
            input=sample_chat_workflow_state, agent_name="test_agent"
        )

    def test_get_model(self, mock_prompt):
        adapter = DefaultPromptAdapter(mock_prompt)
        result = adapter.get_model()

        assert result == mock_prompt.model


@pytest.mark.parametrize(
    "prompt_template",
    [
        {
            "system_static": "<model_info>\n<model_name>{{ model_friendly_name }}</model_name>\n</model_info>",
            "user": "{{ message.content }}",
        }
    ],
)
class TestPromptAdapterFriendlyName:
    """Test friendly_name functionality in prompt adapters."""

    @pytest.fixture
    def mock_gitlab_instance_info(self):
        """Mock GitLab instance info."""
        return GitLabInstanceInfo(
            instance_type="gitlab-com",
            instance_url="https://gitlab.com",
            instance_version="17.0.0",
        )

    @patch(
        "duo_workflow_service.gitlab.gitlab_service_context.GitLabServiceContext.get_current_instance_info"
    )
    @patch("duo_workflow_service.agents.prompt_adapter.current_model_metadata_context")
    def test_chat_agent_prompt_template_injects_friendly_name(
        self,
        mock_context,
        mock_instance_info,
        mock_gitlab_instance_info,
        model_metadata: ModelMetadata,
        project,
        namespace,
        prompt_config,
    ):
        """Test that ChatAgentPromptTemplate injects friendly_name into template."""
        mock_context.get.return_value = model_metadata
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_config)

        input_data: ChatWorkflowState = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": project,
            "namespace": namespace,
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "goal": "Test goal",
            "approval": None,
            "preapproved_tools": None,
        }

        with patch(
            "duo_workflow_service.agents.prompt_adapter.jinja2_formatter"
        ) as mock_formatter:
            mock_formatter.return_value = "Formatted prompt"

            adapter.invoke(input_data, agent_name="test_agent")

            call_args_list = mock_formatter.call_args_list

            _, first_kwargs = call_args_list[0]
            assert "model_friendly_name" in first_kwargs
            assert first_kwargs["model_friendly_name"] == model_metadata.friendly_name

    @patch(
        "duo_workflow_service.gitlab.gitlab_service_context.GitLabServiceContext.get_current_instance_info"
    )
    @patch("duo_workflow_service.agents.prompt_adapter.current_model_metadata_context")
    def test_chat_agent_prompt_template_fallback_when_no_model_metadata(
        self,
        mock_context,
        mock_instance_info,
        mock_gitlab_instance_info,
        prompt_config,
    ):
        """Test that ChatAgentPromptTemplate handles missing model metadata gracefully."""
        mock_context.get.return_value = None
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_config)

        input_data = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": Project(id=1, name="test"),
            "namespace": Namespace(id=1, name="test"),
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "goal": "Test goal",
            "approval": None,
        }

        with patch(
            "duo_workflow_service.agents.prompt_adapter.jinja2_formatter"
        ) as mock_formatter:
            mock_formatter.return_value = "Formatted prompt"

            adapter.invoke(input_data, agent_name="test_agent")

            call_args_list = mock_formatter.call_args_list

            _, first_kwargs = call_args_list[0]
            assert first_kwargs["model_friendly_name"] == "Unknown"

    @patch(
        "duo_workflow_service.gitlab.gitlab_service_context.GitLabServiceContext.get_current_instance_info"
    )
    @patch("duo_workflow_service.agents.prompt_adapter.current_model_metadata_context")
    def test_chat_agent_prompt_template_fallback_when_no_friendly_name(
        self,
        mock_context,
        mock_instance_info,
        mock_gitlab_instance_info,
        prompt_config,
        model_metadata: ModelMetadata,
        project,
        namespace,
    ):
        """Test ChatAgentPromptTemplate when model metadata exists but has no friendly_name."""
        model_metadata.friendly_name = None

        # Setup mocks
        mock_context.get.return_value = model_metadata
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_config)

        input_data: ChatWorkflowState = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": project,
            "namespace": namespace,
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "goal": "Test goal",
            "approval": None,
            "preapproved_tools": None,
        }

        with patch(
            "duo_workflow_service.agents.prompt_adapter.jinja2_formatter"
        ) as mock_formatter:
            mock_formatter.return_value = "Formatted prompt"

            adapter.invoke(input_data, agent_name="test_agent")

            call_args_list = mock_formatter.call_args_list

            _, first_kwargs = call_args_list[0]
            assert first_kwargs["model_friendly_name"] == "Unknown"
