from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.prompt_adapter import ChatAgentPromptTemplate
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.entities.state import (
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
)
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from lib.internal_events.event_enum import CategoryEnum


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


@pytest.fixture(name="workflow_type")
def workflow_type_fixture() -> str:
    return CategoryEnum.WORKFLOW_CHAT.value


@pytest.fixture(name="chat_agent")
def chat_agent_fixture(system_template_override: str):
    mock_prompt_adapter = Mock()
    mock_prompt_adapter.get_response.return_value = AIMessage(content="Hello there!")
    mock_prompt_adapter.get_model.return_value = Mock()
    mock_tools_registry = Mock(spec=ToolsRegistry)
    yield ChatAgent(
        name="Chat Agent",
        prompt_adapter=mock_prompt_adapter,
        tools_registry=mock_tools_registry,
        system_template_override=system_template_override,
    )


@pytest.fixture(autouse=True)
def prepare_container(
    mock_duo_workflow_service_container,
):
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
    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=AIMessage(content="Hello there!")
    )

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
            message_id=None,
        )
    ]
    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED


class TestChatAgentToolCallMessageOrdering:
    """Test the tool call message ordering fix."""

    @pytest.mark.asyncio
    async def test_tool_call_followed_by_human_message_inserts_tool_results(
        self, chat_agent, system_template_override
    ):
        """Test that tool calls followed by human messages get tool results inserted."""
        ai_message_with_tool_call = AIMessage(
            content="I'll help you with that.",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "id": "call_123",
                    "type": "tool_call",
                },
                {
                    "name": "another_tool",
                    "args": {"param2": "value2"},
                    "id": "call_456",
                    "type": "tool_call",
                },
            ],
        )

        human_followup = HumanMessage(content="Actually, let me clarify something.")

        input_with_tool_call_issue = {
            "conversation_history": {
                "Chat Agent": [
                    HumanMessage(content="Can you help me?"),
                    ai_message_with_tool_call,
                    human_followup,
                ]
            },
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "project": None,
            "namespace": None,
            "approval": None,
        }

        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(content="I understand your clarification.")
        )

        result = await chat_agent.run(input_with_tool_call_issue)

        conversation_history = result["conversation_history"]["Chat Agent"]

        assert len(conversation_history) == 1

        chat_agent.prompt_adapter.get_response.assert_called_once_with(
            {
                "conversation_history": {
                    "Chat Agent": [
                        HumanMessage(
                            content="Can you help me?",
                            additional_kwargs={},
                            response_metadata={},
                        ),
                        AIMessage(
                            content="I'll help you with that.",
                            additional_kwargs={},
                            response_metadata={},
                            tool_calls=[
                                {
                                    "name": "test_tool",
                                    "args": {"param": "value"},
                                    "id": "call_123",
                                    "type": "tool_call",
                                },
                                {
                                    "name": "another_tool",
                                    "args": {"param2": "value2"},
                                    "id": "call_456",
                                    "type": "tool_call",
                                },
                            ],
                        ),
                        ToolMessage(
                            content="Tool is cancelled and a user will provide a follow up message.",
                            tool_call_id="call_123",
                        ),
                        ToolMessage(
                            content="Tool is cancelled and a user will provide a follow up message.",
                            tool_call_id="call_456",
                        ),
                        HumanMessage(
                            content="Actually, let me clarify something.",
                            additional_kwargs={},
                            response_metadata={},
                        ),
                    ]
                },
                "plan": {"steps": []},
                "status": WorkflowStatusEnum.EXECUTION,
                "ui_chat_log": [],
                "last_human_input": None,
                "project": None,
                "namespace": None,
                "approval": None,
            },
            system_template_override=system_template_override,
        )

    @pytest.mark.asyncio
    async def test_normal_conversation_flow_unchanged(self, chat_agent):
        """Test that normal conversation flows are not affected by the fix."""
        input_normal_flow = {
            "conversation_history": {
                "Chat Agent": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                    HumanMessage(content="How are you?"),
                ]
            },
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "project": None,
            "namespace": None,
            "approval": None,
        }

        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(content="I'm doing well!")
        )

        await chat_agent.run(input_normal_flow)

        called_input = chat_agent.prompt_adapter.get_response.call_args[0][0]
        original_history = called_input["conversation_history"]["Chat Agent"]

        assert len(original_history) == 3
        assert isinstance(original_history[0], HumanMessage)
        assert isinstance(original_history[1], AIMessage)
        assert isinstance(original_history[2], HumanMessage)

        assert (
            not hasattr(original_history[1], "tool_calls")
            or not original_history[1].tool_calls
        )

    @pytest.mark.asyncio
    async def test_handle_wrong_messages_order_missing_conversation_history_key(
        self, chat_agent
    ):
        input_missing_agent_key = {
            "conversation_history": {},  # Empty - no "Chat Agent" key
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "project": None,
            "namespace": None,
            "approval": None,
        }

        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(content="Hello there!")
        )

        # This should not raise a KeyError
        result = await chat_agent.run(input_missing_agent_key)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        chat_agent.prompt_adapter.get_response.assert_called_once()


@pytest.mark.asyncio
async def test_chat_agent_generic_error_handling(chat_agent, input):
    """Test that ChatAgent properly handles generic exceptions."""
    chat_agent.prompt_adapter.get_response = AsyncMock(
        side_effect=Exception("Test generic error")
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
    # pylint: disable=line-too-long
    assert (
        result["ui_chat_log"][0]["content"]
        == "There was an error processing your request in the Duo Agent Platform, please try again or contact support if the issue persists."
    )


@pytest.mark.asyncio
async def test_chat_agent_provider_4xx_error_handling(chat_agent, input):
    chat_agent.prompt_adapter.get_response = AsyncMock(
        side_effect=APIStatusError(
            message="Test API error",
            response=Mock(status_code=400),
            body={"error": {"message": "Bad request"}},
        )
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
    # pylint: disable=line-too-long
    assert (
        result["ui_chat_log"][0]["content"]
        == "There was an error processing your request in the Duo Agent Platform, please try again or contact support if the issue persists."
    )


@pytest.mark.asyncio
async def test_chat_agent_provider_5xx_error_handling(chat_agent, input):
    """Test that ChatAgent properly handles APIStatusError exceptions."""
    # Mock the prompt adapter to raise an APIStatusError
    chat_agent.prompt_adapter.get_response = AsyncMock(
        side_effect=APIStatusError(
            message="Test API error",
            response=Mock(status_code=500),
            body={"error": {"message": "Internal server error"}},
        )
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
    # pylint: disable=line-too-long
    assert (
        result["ui_chat_log"][0]["content"]
        == "There was an error connecting to the chosen LLM provider, please try again or contact support if the issue persists."
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
    ],
)
class TestChatAgentGitLabInstanceInfo:
    """Test GitLab instance info integration with ChatAgent static prompt."""

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
        self, prompt_config, input_with_project
    ):
        """Test static prompt contains correct GitLab instance info from context."""
        template = ChatAgentPromptTemplate(prompt_config)

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
        self, prompt_config, input_with_project
    ):
        """Test static prompt handles missing GitLab context gracefully."""
        template = ChatAgentPromptTemplate(prompt_config)

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
async def test_agentic_fake_model_bypasses_tool_approval(input):
    mock_model = Mock()
    mock_model._is_agentic_mock_model = True

    mock_prompt_adapter = Mock()
    mock_prompt_adapter.get_model.return_value = mock_model

    mock_tools_registry = Mock(spec=ToolsRegistry)
    mock_tools_registry.approval_required.return_value = True

    chat_agent = ChatAgent(
        name="Chat Agent",
        prompt_adapter=mock_prompt_adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
    )

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

    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=ai_message_with_tools
    )

    result = await chat_agent.run(input)

    assert result["status"] == WorkflowStatusEnum.EXECUTION
