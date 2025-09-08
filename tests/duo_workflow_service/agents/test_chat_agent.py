from datetime import datetime, timezone
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, call, patch

import pytest
from anthropic import APIStatusError
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.prompt_values import ChatPromptValue

from ai_gateway.container import ContainerApplication
from ai_gateway.models.agentic_mock import AgenticFakeModel
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent, ChatAgentPromptTemplate
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.entities.state import (
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
)
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum


@pytest.fixture(name="mock_datetime")
def mock_datetime_fixture(mock_now: datetime):
    with patch("duo_workflow_service.agents.chat_agent.datetime") as mock:
        mock.now.return_value = mock_now
        mock.timezone = timezone
        yield mock


@pytest.fixture(name="config_values")
def config_values_fixture():
    return {"mock_model_responses": True}


@pytest.fixture(name="user_is_debug")
def user_is_debug_fixture():
    return True


@pytest.fixture(name="prompt_name")
def prompt_name_fixture():
    return "Chat Agent"


@pytest.fixture(name="chat_agent")
def chat_agent_fixture(model_factory, prompt_config, model_metadata):
    yield ChatAgent(
        model_factory=model_factory, config=prompt_config, model_metadata=model_metadata
    )


@pytest.fixture(autouse=True)
def prepare_container(
    mock_duo_workflow_service_container,
):  # pylint: disable=unused-argument
    mock_duo_workflow_service_container.wire(
        modules=["tests.duo_workflow_service.agents.test_chat_agent"]
    )


@pytest.fixture(name="input")
def input_fixture():
    return {
        "conversation_history": {"Chat Agent": [HumanMessage(content="hi")]},
        "plan": {"steps": []},
        "status": WorkflowStatusEnum.EXECUTION,
        "ui_chat_log": [],
        "last_human_input": None,
        "project": None,
        "namespace": None,
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


@pytest.mark.asyncio
@inject
async def test_template_with_project(
    input,
    user: CloudConnectorUser,
    prompt_registry: LocalPromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
):
    input["project"] = Project(
        id=1,
        name="gitlab project",
        web_url="https://gitlab.com/gitlab-org/gitlab",
        description="awesome project",
        http_url_to_repo="",
        default_branch=None,
        languages=[],
        exclusion_rules=None,
    )

    chat_agent: ChatAgent = prompt_registry.get_on_behalf(  # type: ignore[assignment]
        user=user,
        prompt_id="chat/agent",
        prompt_version="^1.0.0",
        model_metadata=None,
        internal_event_category=__name__,
        tools=None,
    )

    result: Any = await chat_agent.prompt_tpl.ainvoke(input, agent_name=chat_agent.name)

    assert isinstance(result.messages[1], SystemMessage)
    assert "<project_id>1</project_id>" in result.messages[1].content
    assert "<project_name>gitlab project</project_name>" in result.messages[1].content
    assert (
        "<project_url>https://gitlab.com/gitlab-org/gitlab</project_url>"
        in result.messages[1].content
    )
    assert "<namespace>" not in result.messages[1].content


@pytest.mark.asyncio
@inject
async def test_template_with_namespace(
    input,
    user: CloudConnectorUser,
    prompt_registry: LocalPromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
):
    input["namespace"] = Namespace(
        id=1,
        name="gitlab-org",
        web_url="https://gitlab.com/gitlab-org",
        description="awesome organization",
    )
    chat_agent: ChatAgent = prompt_registry.get_on_behalf(  # type: ignore[assignment]
        user=user,
        prompt_id="chat/agent",
        prompt_version="^1.0.0",
        model_metadata=None,
        internal_event_category=__name__,
        tools=None,
    )

    result: Any = await chat_agent.prompt_tpl.ainvoke(input, agent_name=chat_agent.name)

    assert isinstance(result.messages[1], SystemMessage)
    assert "<project>" not in result.messages[1].content
    assert "<namespace_id>1</namespace_id>" in result.messages[1].content
    assert (
        "<namespace_description>awesome organization</namespace_description>"
        in result.messages[1].content
    )
    assert "<namespace_name>gitlab-org</namespace_name>" in result.messages[1].content
    assert (
        "<namespace_url>https://gitlab.com/gitlab-org</namespace_url>"
        in result.messages[1].content
    )


class TestChatAgentTrackTokensData:
    @pytest.fixture(name="unit_primitives")
    def unit_primitives_fixture(self):
        return ["duo_chat"]

    @pytest.fixture(name="usage_metadata")
    def usage_metadata_fixture(self):
        return UsageMetadata(input_tokens=1, output_tokens=2, total_tokens=3)

    @pytest.mark.asyncio
    async def test_track_tokens_data(
        self, chat_agent, input, internal_event_client: Mock
    ):
        chat_agent.internal_event_client = internal_event_client

        if hasattr(chat_agent.model, "usage_metadata"):
            chat_agent.model.usage_metadata = UsageMetadata(
                input_tokens=1, output_tokens=2, total_tokens=3
            )

        await chat_agent.run(input)

        assert internal_event_client.track_event.call_count == 2
        # Get all calls and sort them by event type for consistent testing
        calls = internal_event_client.track_event.call_args_list

        # Find the base class token usage call
        base_call = None
        agent_call = None

        for call_obj in calls:
            args, kwargs = call_obj
            if args and args[0] == "token_usage_duo_chat":
                base_call = call_obj
            elif (
                "event_name" in kwargs
                and kwargs["event_name"] == EventEnum.TOKEN_PER_USER_PROMPT.value
            ):
                agent_call = call_obj

        # Verify base class call (from handle_usage_metadata)
        assert base_call == call(
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

        # Verify agent-specific call (from _track_tokens_data)
        assert agent_call == call(
            event_name=EventEnum.TOKEN_PER_USER_PROMPT.value,
            additional_properties=InternalEventAdditionalProperties(
                label="Chat Agent",
                property=EventPropertyEnum.WORKFLOW_ID.value,
                value=ANY,
                input_tokens=1,
                output_tokens=2,
                total_tokens=3,
            ),
            category=CategoryEnum.WORKFLOW_CHAT.value,
        )


class TestChatAgentPromptTemplate:
    @pytest.fixture(name="prompt_template_with_split_system")
    def prompt_template_with_split_system_fixture(self):
        """Prompt template with both system_static and system_dynamic parts."""
        return {
            "system_static": """You are GitLab Duo Chat, an AI coding assistant.

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
</context>""",
            "user": "{{ message.content }}",
        }

    @pytest.fixture(name="chat_workflow_state")
    def chat_workflow_state_fixture(self):
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

        # Test without GitLab context (should use fallback values)
        result = template.invoke(
            chat_workflow_state,
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

        result = template.invoke(
            state_without_project, agent_name="test_agent", is_anthropic_model=False
        )

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

        result = template.invoke(
            chat_workflow_state, agent_name="test_agent", is_anthropic_model=False
        )

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

        result = template.invoke(
            state_with_history, agent_name="test_agent", is_anthropic_model=False
        )

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
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        result = template.invoke(
            chat_workflow_state, agent_name="test_agent", is_anthropic_model=True
        )

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
        assert content_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

        # Dynamic system message should remain as plain text
        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message, SystemMessage)
        assert isinstance(dynamic_system_message.content, str)

    def test_anthropic_cache_control_disabled(
        self,
        prompt_template_with_split_system,
        chat_workflow_state,
    ):
        """Test that cache_control is NOT added when model is not Anthropic."""
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        result = template.invoke(
            chat_workflow_state, agent_name="test_agent", is_anthropic_model=False
        )

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
        template = ChatAgentPromptTemplate(prompt_template_with_split_system)

        result = template.invoke(
            chat_workflow_state, agent_name="test_agent", is_anthropic_model=True
        )

        messages = result.messages

        # Static system message should have cache_control
        static_system_message = messages[0]
        assert isinstance(static_system_message.content, list)
        assert static_system_message.content[0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

        # Dynamic system message should NOT have cache_control
        dynamic_system_message = messages[1]
        assert isinstance(dynamic_system_message.content, str)
        # Verify it's the dynamic content by checking for date
        expected_date = mock_datetime.now().strftime("%Y-%m-%d")
        assert expected_date in dynamic_system_message.content


@pytest.mark.asyncio
async def test_chat_agent_api_error_handling(chat_agent, input):
    """Test that ChatAgent properly handles APIStatusError exceptions."""
    # Mock the superclass ainvoke method to raise an APIStatusError
    with patch.object(
        chat_agent.__class__.__bases__[0], "ainvoke", new_callable=AsyncMock
    ) as mock_ainvoke:
        mock_ainvoke.side_effect = APIStatusError(
            message="Test API error",
            response=Mock(status_code=500),
            body={"error": {"message": "Internal server error"}},
        )

        result = await chat_agent.run(input)

        # Verify error response structure
        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert "conversation_history" in result
        assert result["conversation_history"]["Chat Agent"][0].content.startswith(
            "There was an error processing your request:"
        )
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["status"] == ToolStatus.FAILURE
        assert (
            result["ui_chat_log"][0]["content"]
            == "There was an error processing your request. Please try again or contact support if the issue persists."
        )


class TestChatAgentGitLabInstanceInfo:
    """Test GitLab instance info integration with ChatAgent static prompt."""

    @pytest.fixture(name="prompt_template_with_gitlab_info")
    def prompt_template_with_gitlab_info_fixture(self):
        """Prompt template that includes GitLab instance info in static system prompt."""
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
The current date is {{ current_date }}.
{%- if project %}
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
</namespace>
{%- endif %}
</context>""",
            "user": "{{ message.content }}",
        }

    @pytest.fixture(name="input_with_project")
    def input_with_project_fixture(self):
        """Input with project data."""
        return ChatWorkflowState(
            plan={"steps": []},
            status="execution",
            conversation_history={"test_agent": [HumanMessage(content="Hello")]},
            ui_chat_log=[],
            last_human_input=None,
            project=Project(
                id=123,
                name="test-project",
                description="Test project",
                http_url_to_repo="https://gitlab.com/test/project.git",
                web_url="https://gitlab.com/test/project",
                default_branch="main",
                languages=[],
                exclusion_rules=[],
            ),
            namespace=None,
            approval=None,
        )

    def test_static_prompt_contains_gitlab_instance_info(
        self, prompt_template_with_gitlab_info, input_with_project
    ):
        """Test static prompt contains correct GitLab instance info from context."""
        template = ChatAgentPromptTemplate(prompt_template_with_gitlab_info)

        # Mock the GitLab instance info service
        mock_gitlab_service = Mock()
        mock_gitlab_info = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0-ee",
        )
        mock_gitlab_service.create_from_project_and_namespace.return_value = (
            mock_gitlab_info
        )

        # Use the context manager to provide GitLab info
        with GitLabServiceContext(
            mock_gitlab_service,
            project=input_with_project["project"],
            namespace=input_with_project["namespace"],
        ):
            result = template.invoke(
                input_with_project,
                agent_name="test_agent",
                is_anthropic_model=False,
            )

        messages = result.messages
        assert len(messages) == 3  # static system, dynamic system, user

        # Check static system message contains GitLab instance info
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)
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

        # Verify service was called with correct parameters
        mock_gitlab_service.create_from_project_and_namespace.assert_called_once_with(
            input_with_project["project"], input_with_project["namespace"]
        )

    def test_static_prompt_without_gitlab_context(
        self, prompt_template_with_gitlab_info, input_with_project
    ):
        """Test static prompt handles missing GitLab context gracefully."""
        template = ChatAgentPromptTemplate(prompt_template_with_gitlab_info)

        # Call template without GitLab context
        result = template.invoke(
            input_with_project,
            agent_name="test_agent",
            is_anthropic_model=False,
        )

        messages = result.messages
        assert len(messages) == 3  # static system, dynamic system, user

        # Check static system message contains fallback "Unknown" values
        static_system_message = messages[0]
        assert isinstance(static_system_message, SystemMessage)
        assert (
            "<gitlab_instance_type>Unknown</gitlab_instance_type>"
            in static_system_message.content
        )
        assert (
            "<gitlab_instance_url>Unknown</gitlab_instance_url>"
            in static_system_message.content
        )
        assert (
            "<gitlab_instance_version>Unknown</gitlab_instance_version>"
            in static_system_message.content
        )


@pytest.mark.asyncio
async def test_agentic_fake_model_bypasses_tool_approval(
    prompt_config, model_metadata, input
):
    def agentic_model_factory(
        *, model: str, **kwargs
    ):  # pylint: disable=unused-argument
        return AgenticFakeModel()

    chat_agent = ChatAgent(
        model_factory=agentic_model_factory,
        config=prompt_config,
        model_metadata=model_metadata,
    )

    chat_agent.tools_registry = Mock(spec=ToolsRegistry)
    chat_agent.tools_registry.approval_required.return_value = True

    # Create an AI message with tool calls to simulate what would happen
    ai_message_with_tools = AIMessage(
        content="I need to use a tool",
        tool_calls=[
            {
                "name": "test_tool",
                "args": {"param": "value"},
                "id": "call_123",
                "type": "tool_call",
            }
        ],
    )

    # Mock the agent response to return our AI message with tools
    with patch.object(
        chat_agent.__class__.__bases__[0], "ainvoke", new_callable=AsyncMock
    ) as mock_ainvoke:
        mock_ainvoke.return_value = ai_message_with_tools

        result = await chat_agent.run(input)

        assert result["status"] == WorkflowStatusEnum.EXECUTION
