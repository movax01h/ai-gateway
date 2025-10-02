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
    CustomPromptAdapter,
    DefaultPromptAdapter,
    create_adapter,
)
from duo_workflow_service.entities.state import ChatWorkflowState
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


@pytest.fixture(name="sample_chat_workflow_state")
def sample_chat_workflow_state_fixture():
    """Sample ChatWorkflowState for testing."""
    return ChatWorkflowState(
        plan={"steps": []},
        status="execution",
        conversation_history={"test_agent": [HumanMessage(content="Hello")]},
        ui_chat_log=[],
        last_human_input=None,
        goal="Test goal",
        project=Project(
            id=123,
            name="Test Project",
            description="Test project description",
            http_url_to_repo="https://gitlab.com/test/project.git",
            web_url="https://gitlab.com/test/project",
            default_branch="main",
            languages=[],
            exclusion_rules=[],
        ),
        namespace=Namespace(
            id=456,
            name="test-org",
            description="Test organization",
            web_url="https://gitlab.com/test-org",
        ),
        approval=None,
    )


@pytest.fixture(name="prompt_template_with_split_system")
def prompt_template_with_split_system_fixture():
    """Prompt template with both system_static and system_dynamic parts."""
    return {
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
The current date is {{ current_date }}. The current time is {{ current_time }}. The user's timezone is
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


class TestChatAgentPromptTemplate:
    def test_split_system_prompts_create_separate_messages(
        self,
        prompt_template_with_split_system,
        sample_chat_workflow_state,
        mock_datetime,
    ):
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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
                is_anthropic_model=False,
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
        assert "<namespace_id>456</namespace_id>" in dynamic_system_message.content
        assert (
            "<namespace_name>test-org</namespace_name>"
            in dynamic_system_message.content
        )

        # Third message should be user message
        user_message = messages[2]
        assert isinstance(user_message, HumanMessage)

    def test_conversation_history_processing(self, prompt_template_with_split_system):
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

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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

    @pytest.mark.parametrize(
        "is_anthropic_model,expected_content_type",
        [
            (True, list),
            (False, str),
        ],
        ids=[
            "cache_control_enabled",
            "cache_control_disabled",
        ],
    )
    def test_anthropic_cache_control(
        self,
        prompt_template_with_split_system,
        sample_chat_workflow_state,
        is_anthropic_model,
        expected_content_type,
    ):
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        with patch.object(
            GitLabServiceContext, "get_current_instance_info", return_value=None
        ):
            result = template.invoke(
                sample_chat_workflow_state,
                agent_name="test_agent",
                is_anthropic_model=is_anthropic_model,
            )

        assert isinstance(result, ChatPromptValue)
        messages = result.messages

        # Should have 2 system messages and 1 user message
        assert len(messages) == 3

        # Check the static system message
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)
        assert isinstance(static_system_message.content, expected_content_type)

        if is_anthropic_model:
            assert len(static_system_message.content) == 1
            content_block = static_system_message.content[0]
            assert isinstance(content_block, dict)
            assert content_block["type"] == "text"
            assert "GitLab Duo Chat" in content_block["text"]
            assert "<core_mission>" in content_block["text"]
            assert content_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

            dynamic_system_message = messages[1]
            assert isinstance(dynamic_system_message, SystemMessage)
            assert isinstance(dynamic_system_message.content, str)
        else:
            assert "GitLab Duo Chat" in static_system_message.content

    def test_slash_command_parsing(self, prompt_template_with_split_system):
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

        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

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

        mock_prompt_value = Mock()
        mock_prompt_value.to_messages.return_value = [HumanMessage(content="test")]
        mock_prompt.prompt_tpl.invoke.return_value = mock_prompt_value

        expected_response = AIMessage(content="Test response")
        mock_prompt.model.ainvoke.return_value = expected_response

        result = await adapter.get_response(sample_chat_workflow_state)

        assert result == expected_response
        mock_prompt.prompt_tpl.invoke.assert_called_once_with(
            input=sample_chat_workflow_state,
            agent_name="test_agent",
            is_anthropic_model=True,
        )
        mock_prompt.model.ainvoke.assert_called_once()

    def test_get_model(self, mock_prompt):
        adapter = DefaultPromptAdapter(mock_prompt)
        result = adapter.get_model()

        assert result == mock_prompt.model


class TestCustomPromptAdapter:
    @pytest.fixture(name="mock_prompt")
    def mock_prompt_fixture(self):
        prompt = Mock(spec=Prompt)
        prompt.name = "test_agent"
        prompt.model = AsyncMock()
        prompt.ainvoke = AsyncMock()
        return prompt

    @pytest.mark.asyncio
    async def test_get_response(
        self, mock_prompt, sample_chat_workflow_state, mock_datetime
    ):
        adapter = CustomPromptAdapter(mock_prompt)

        expected_response = AIMessage(content="Custom response")
        mock_prompt.ainvoke.return_value = expected_response

        result = await adapter.get_response(sample_chat_workflow_state)

        assert result == expected_response

        # Verify the call was made with correct variables
        mock_prompt.ainvoke.assert_called_once()
        call_args = mock_prompt.ainvoke.call_args[1]["input"]

        assert call_args["goal"] == sample_chat_workflow_state["goal"]
        assert call_args["project"] == sample_chat_workflow_state["project"]
        assert call_args["namespace"] == sample_chat_workflow_state["namespace"]
        assert (
            call_args["history"]
            == sample_chat_workflow_state["conversation_history"]["test_agent"]
        )
        assert call_args["current_date"] == mock_datetime.now().strftime("%Y-%m-%d")
        assert call_args["current_time"] == mock_datetime.now().strftime("%H:%M:%S")
        assert (
            call_args["current_timezone"] == mock_datetime.now().astimezone().tzname()
        )

    def test_get_model(self, mock_prompt):
        adapter = CustomPromptAdapter(mock_prompt)
        result = adapter.get_model()

        assert result == mock_prompt.model


class TestCreateAdapter:
    @pytest.fixture(name="mock_prompt")
    def mock_prompt_fixture(self):
        prompt = Mock(spec=Prompt)
        prompt.name = "test_agent"
        prompt.prompt_tpl = Mock()
        return prompt

    @pytest.mark.parametrize(
        "use_custom_adapter,expected_adapter_type,expected_attribute",
        [
            (False, DefaultPromptAdapter, "_base_prompt"),
            (True, CustomPromptAdapter, "_prompt"),
            (None, DefaultPromptAdapter, "_base_prompt"),  # default when not specified
        ],
        ids=[
            "default_adapter",
            "custom_adapter",
            "default_when_not_specified",
        ],
    )
    def test_create_adapter(
        self, mock_prompt, use_custom_adapter, expected_adapter_type, expected_attribute
    ):
        kwargs = {"prompt": mock_prompt}
        if use_custom_adapter is not None:
            kwargs["use_custom_adapter"] = use_custom_adapter

        adapter = create_adapter(**kwargs)

        assert isinstance(adapter, expected_adapter_type)
        assert getattr(adapter, expected_attribute) == mock_prompt


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

    @pytest.fixture
    def mock_model_metadata(self):
        """Mock model metadata with friendly_name."""
        return ModelMetadata(
            name="claude_sonnet_4_5_20250929",
            provider="gitlab",
            friendly_name="Claude Sonnet 4.5 - Anthropic",
        )

    @pytest.fixture
    def mock_prompt_template(self):
        """Mock prompt template with model metadata injection."""
        mock_prompt = Mock()
        mock_prompt.text = "<model_info>\n<model_name>{{ model_friendly_name }}</model_name>\n</model_info>"
        mock_user_prompt = Mock()
        mock_user_prompt.text = "{{ message.content }}"
        return {"system_static": mock_prompt, "user": mock_user_prompt}

    @patch(
        "duo_workflow_service.gitlab.gitlab_service_context.GitLabServiceContext.get_current_instance_info"
    )
    @patch("duo_workflow_service.agents.prompt_adapter.current_model_metadata_context")
    def test_chat_agent_prompt_template_injects_friendly_name(
        self,
        mock_context,
        mock_instance_info,
        mock_gitlab_instance_info,
        mock_model_metadata,
        mock_prompt_template,
    ):
        """Test that ChatAgentPromptTemplate injects friendly_name into template."""
        mock_context.get.return_value = mock_model_metadata
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_template=mock_prompt_template)

        input_data = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": Project(id=1, name="test"),
            "namespace": Namespace(id=1, name="test"),
            "plan": {"steps": []},
            "status": "execution",
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
            assert "model_friendly_name" in first_kwargs
            assert (
                first_kwargs["model_friendly_name"] == "Claude Sonnet 4.5 - Anthropic"
            )

    @patch(
        "duo_workflow_service.gitlab.gitlab_service_context.GitLabServiceContext.get_current_instance_info"
    )
    @patch("duo_workflow_service.agents.prompt_adapter.current_model_metadata_context")
    def test_chat_agent_prompt_template_fallback_when_no_model_metadata(
        self,
        mock_context,
        mock_instance_info,
        mock_gitlab_instance_info,
        mock_prompt_template,
    ):
        """Test that ChatAgentPromptTemplate handles missing model metadata gracefully."""
        mock_context.get.return_value = None
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_template=mock_prompt_template)

        input_data = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": Project(id=1, name="test"),
            "namespace": Namespace(id=1, name="test"),
            "plan": {"steps": []},
            "status": "execution",
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
        mock_prompt_template,
    ):
        """Test ChatAgentPromptTemplate when model metadata exists but has no friendly_name."""
        metadata_no_friendly_name = ModelMetadata(
            name="test_model",
            provider="test_provider",
            friendly_name=None,
        )

        # Setup mocks
        mock_context.get.return_value = metadata_no_friendly_name
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_template=mock_prompt_template)

        input_data = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": Project(id=1, name="test"),
            "namespace": Namespace(id=1, name="test"),
            "plan": {"steps": []},
            "status": "execution",
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
    def test_chat_agent_prompt_template_with_self_hosted_model(
        self,
        mock_context,
        mock_instance_info,
        mock_gitlab_instance_info,
        mock_prompt_template,
    ):
        """Test ChatAgentPromptTemplate with self-hosted model friendly_name."""
        self_hosted_metadata = ModelMetadata(
            name="llama3",
            provider="custom_openai",
            friendly_name="Llama3",
        )

        mock_context.get.return_value = self_hosted_metadata
        mock_instance_info.return_value = mock_gitlab_instance_info

        adapter = ChatAgentPromptTemplate(prompt_template=mock_prompt_template)

        input_data = {
            "conversation_history": {"test_agent": [HumanMessage(content="Hello")]},
            "project": Project(id=1, name="test"),
            "namespace": Namespace(id=1, name="test"),
            "plan": {"steps": []},
            "status": "execution",
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
            assert first_kwargs["model_friendly_name"] == "Llama3"
