# pylint: disable=too-many-lines
import json
from datetime import datetime, timezone
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from anthropic import APIStatusError
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    InvalidToolCall,
    SystemMessage,
    ToolMessage,
)
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.types import Overwrite

from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agents.chat_agent import ChatAgent, _suggest_patterns
from duo_workflow_service.agents.prompt_adapter import ChatAgentPromptTemplate
from duo_workflow_service.checkpointer.gitlab_workflow import _serialize_channel_blobs
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    build_compaction_tool_card,
)
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionResult,
    OptimizationResult,
)
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.entities.state import (
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    _conversation_history_reducer,
)
from duo_workflow_service.errors.typing import NotifiableException
from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandValidationError,
)
from duo_workflow_service.tools import MalformedToolCallError, Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties
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


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    mock = Mock(spec=Toolset)
    mock.validate_tool_call.return_value = None
    return mock


def _make_passthrough_pipeline(
    results: list[OptimizationResult] | None = None,
) -> HistoryOptimizerPipeline:
    """Build a HistoryOptimizerPipeline mock that returns history unchanged.

    ``results`` overrides the returned results list; default is a single
    ``was_modified=False`` result mirroring a no-op run.
    """
    mock_pipeline = Mock(spec=HistoryOptimizerPipeline)

    async def optimize(history):
        default = [OptimizationResult(messages=history, was_modified=False)]
        return history, results if results is not None else default

    mock_pipeline.optimize = AsyncMock(side_effect=optimize)
    return mock_pipeline


@pytest.fixture(name="chat_agent")
def chat_agent_fixture(system_template_override: str, mock_toolset):
    mock_prompt_adapter = Mock()
    mock_prompt_adapter.get_response.return_value = AIMessage(content="Hello there!")
    mock_prompt_adapter.get_model.return_value = Mock()
    mock_tools_registry = Mock(spec=ToolsRegistry)
    yield ChatAgent(
        name="Chat Agent",
        prompt_adapter=mock_prompt_adapter,
        tools_registry=mock_tools_registry,
        system_template_override=system_template_override,
        toolset=mock_toolset,
        optimizer_pipeline=_make_passthrough_pipeline(),
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
        return_value=AIMessage(content="Hello there!", id="agent-msg-id")
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
            message_id="agent-msg-id",
        )
    ]
    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED


@pytest.mark.asyncio
async def test_run_tags_agent_message_after_tier_access_denied(chat_agent, input):
    tier_denied_payload = json.dumps(
        {
            "error": "tier_access_denied",
            "required_plan": "ultimate",
            "message": "Feature requires Ultimate.",
            "link_url": "https://docs.gitlab.com/user/duo_agent_platform/",
        }
    )
    input["conversation_history"]["Chat Agent"] = [
        HumanMessage(content="list vulnerabilities"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_vulnerabilities",
                    "args": {},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content=tier_denied_payload, tool_call_id="call_1"),
    ]
    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=AIMessage(
            content="This feature requires **GitLab Ultimate**.",
            id="agent-msg-tier",
        )
    )

    result = await chat_agent.run(input)

    assert len(result["ui_chat_log"]) == 1
    log_entry = result["ui_chat_log"][0]
    assert log_entry["message_type"] == MessageTypeEnum.AGENT
    assert log_entry["message_sub_type"] == "tier_access_denied"
    assert log_entry["required_plan"] == "ultimate"


@pytest.mark.asyncio
async def test_run_does_not_tag_agent_message_without_tier_access_denied(
    chat_agent, input
):
    input["conversation_history"]["Chat Agent"] = [
        HumanMessage(content="list issues"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_issues",
                    "args": {},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content='{"issues": []}', tool_call_id="call_2"),
    ]
    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=AIMessage(content="No issues found.", id="agent-msg-ok")
    )

    result = await chat_agent.run(input)

    assert len(result["ui_chat_log"]) == 1
    log_entry = result["ui_chat_log"][0]
    assert log_entry["message_sub_type"] is None
    assert "required_plan" not in log_entry


@pytest.mark.asyncio
async def test_run_does_not_tag_when_tier_denied_in_prior_turn(chat_agent, input):
    tier_denied_payload = json.dumps(
        {
            "error": "tier_access_denied",
            "required_plan": "ultimate",
        }
    )
    # tier_denied happened in a prior turn; current turn has no tool calls
    input["conversation_history"]["Chat Agent"] = [
        HumanMessage(content="list vulnerabilities"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_vulnerabilities",
                    "args": {},
                    "id": "call_old",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content=tier_denied_payload, tool_call_id="call_old"),
        AIMessage(content="Needs Ultimate.", id="agent-prior"),
        HumanMessage(content="ok thanks, anything else?"),
    ]
    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=AIMessage(content="Sure, ask away.", id="agent-msg-followup")
    )

    result = await chat_agent.run(input)

    assert len(result["ui_chat_log"]) == 1
    log_entry = result["ui_chat_log"][0]
    assert log_entry["message_sub_type"] is None
    assert "required_plan" not in log_entry


@pytest.mark.asyncio
async def test_run_tags_agent_message_when_tier_denied_not_last_in_batch(
    chat_agent, input
):
    tier_denied_payload = json.dumps(
        {
            "error": "tier_access_denied",
            "required_plan": "ultimate",
        }
    )
    # Multi-tool batch: tier_denied tool ran before a successful tool in the
    # same turn. The tier_denied ToolMessage is not the last message.
    input["conversation_history"]["Chat Agent"] = [
        HumanMessage(content="list vulnerabilities and list issues"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_vulnerabilities",
                    "args": {},
                    "id": "call_tier",
                    "type": "tool_call",
                },
                {
                    "name": "list_issues",
                    "args": {},
                    "id": "call_ok",
                    "type": "tool_call",
                },
            ],
        ),
        ToolMessage(content=tier_denied_payload, tool_call_id="call_tier"),
        ToolMessage(content='{"issues": []}', tool_call_id="call_ok"),
    ]
    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=AIMessage(
            content="Vulnerabilities require **Ultimate**. No issues found.",
            id="agent-msg-mixed",
        )
    )

    result = await chat_agent.run(input)

    assert len(result["ui_chat_log"]) == 1
    log_entry = result["ui_chat_log"][0]
    assert log_entry["message_type"] == MessageTypeEnum.AGENT
    assert log_entry["message_sub_type"] == "tier_access_denied"
    assert log_entry["required_plan"] == "ultimate"


@pytest.mark.asyncio
async def test_run_does_not_tag_when_tier_denied_content_is_malformed_json(
    chat_agent, input
):
    input["conversation_history"]["Chat Agent"] = [
        HumanMessage(content="list vulnerabilities"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "list_vulnerabilities",
                    "args": {},
                    "id": "call_bad",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(
            content="tier_access_denied: not valid json",
            tool_call_id="call_bad",
        ),
    ]
    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=AIMessage(content="Something went wrong.", id="agent-msg-bad")
    )

    result = await chat_agent.run(input)

    assert len(result["ui_chat_log"]) == 1
    log_entry = result["ui_chat_log"][0]
    assert log_entry["message_sub_type"] is None
    assert "required_plan" not in log_entry


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

        # The reorder is a rewrite, so it comes back as an Overwrite channel
        # update; the input state stays untouched (gitlab-org/gitlab#623342).
        history_update = result["conversation_history"]
        assert isinstance(history_update, Overwrite)
        reordered = history_update.value["Chat Agent"]
        assert isinstance(reordered[2], ToolMessage)
        assert isinstance(reordered[3], ToolMessage)
        assert reordered[-1].content == "I understand your clarification."
        assert input_with_tool_call_issue["conversation_history"]["Chat Agent"] == [
            HumanMessage(content="Can you help me?"),
            ai_message_with_tool_call,
            human_followup,
        ]

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
    original_error = Exception("Test generic error")
    chat_agent.prompt_adapter.get_response = AsyncMock(side_effect=original_error)

    with pytest.raises(NotifiableException) as exc_info:
        await chat_agent.run(input)

    assert str(exc_info.value) == (
        "There was an error processing your request in the Duo Agent Platform, please contact support "
        "if the issue persists."
    )
    assert exc_info.value.__cause__ is original_error


@pytest.mark.asyncio
async def test_chat_agent_provider_4xx_error_handling(chat_agent, input):
    original_error = APIStatusError(
        message="Test API error",
        response=Mock(status_code=400),
        body={"error": {"message": "Bad request"}},
    )
    chat_agent.prompt_adapter.get_response = AsyncMock(side_effect=original_error)

    with pytest.raises(NotifiableException) as exc_info:
        await chat_agent.run(input)

    assert str(exc_info.value) == (
        "There was an error processing your request in the Duo Agent Platform, please contact support "
        "if the issue persists."
    )
    assert exc_info.value.__cause__ is original_error


@pytest.mark.asyncio
async def test_chat_agent_provider_5xx_error_handling(chat_agent, input):
    """Test that ChatAgent properly handles APIStatusError exceptions."""
    original_error = APIStatusError(
        message="Test API error",
        response=Mock(status_code=500),
        body={"error": {"message": "Internal server error"}},
    )
    chat_agent.prompt_adapter.get_response = AsyncMock(side_effect=original_error)

    with pytest.raises(NotifiableException) as exc_info:
        await chat_agent.run(input)

    assert str(exc_info.value) == (
        "There was an error connecting to the chosen LLM provider, please contact support "
        "if the issue persists."
    )
    assert exc_info.value.__cause__ is original_error


@pytest.mark.asyncio
@patch("duo_workflow_service.agents.chat_agent.log_exception")
async def test_chat_agent_invalid_slash_command_error_handling(
    mock_log_exception, chat_agent, input
):
    """Test that ChatAgent properly handles SlashCommandValidationError and returns a user-friendly message."""

    # Mock the prompt adapter to raise a SlashCommandValidationError
    chat_agent.prompt_adapter.get_response = AsyncMock(
        side_effect=SlashCommandValidationError(
            "The command '/invalid_command' does not exist."
        )
    )

    result = await chat_agent.run(input)

    # Verify that log_exception was called with the correct parameters
    mock_log_exception.assert_called_once()
    call_args = mock_log_exception.call_args
    assert isinstance(call_args[0][0], SlashCommandValidationError)
    assert str(call_args[0][0]) == "The command '/invalid_command' does not exist."
    assert call_args[1]["extra"] == {
        "context": "User provided an invalid slash command"
    }

    # Verify error response structure
    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert "conversation_history" in result

    # Check that the error message is returned to the user
    conversation_history = result["conversation_history"]["Chat Agent"]
    assert len(conversation_history) == 1
    assert isinstance(conversation_history[0], AIMessage)
    assert (
        conversation_history[0].content
        == "The command '/invalid_command' does not exist."
    )

    # Check UI chat log
    assert len(result["ui_chat_log"]) == 1
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    assert result["ui_chat_log"][0]["message_id"].startswith("error-")
    assert result["ui_chat_log"][0]["status"] == ToolStatus.FAILURE
    assert (
        result["ui_chat_log"][0]["content"]
        == "The command '/invalid_command' does not exist."
    )


@pytest.mark.asyncio
async def test_chat_agent_invalid_tool_calls_handling(chat_agent, input):
    """Test that ChatAgent passes through invalid tool calls to the workflow for ToolsExecutor to handle."""
    # Create an AIMessage with invalid tool calls
    invalid_tool_calls = [
        InvalidToolCall(
            id="invalid-call-1",
            error="JSON parsing error: unexpected token",
            name="invalid_tool",
            args="{}",
            type="invalid_tool_call",
        ),
    ]

    ai_message_with_invalid_calls = AIMessage(
        content="I'll try to use a tool",
        invalid_tool_calls=invalid_tool_calls,
        id="agent-msg-invalid-id",
    )

    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=ai_message_with_invalid_calls
    )

    result = await chat_agent.run(input)

    # Verify the response structure
    # ChatAgent passes through the AIMessage with invalid_tool_calls to conversation_history
    assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
    assert len(result["conversation_history"]["Chat Agent"]) == 1
    assert isinstance(result["conversation_history"]["Chat Agent"][0], AIMessage)
    assert (
        result["conversation_history"]["Chat Agent"][0].content
        == "I'll try to use a tool"
    )
    assert (
        result["conversation_history"]["Chat Agent"][0].invalid_tool_calls
        == invalid_tool_calls
    )

    # Verify UI chat log contains agent message
    assert len(result["ui_chat_log"]) == 1

    # Verify agent message
    assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    assert result["ui_chat_log"][0]["content"] == "I'll try to use a tool"
    assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS
    assert result["ui_chat_log"][0]["message_id"] == "agent-msg-invalid-id"


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
        self, model_provider, prompt_config, input_with_project
    ):
        """Test static prompt contains correct GitLab instance info from context."""
        template = ChatAgentPromptTemplate(model_provider, prompt_config)

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
        assert (
            len(messages) == 4
        )  # static system, security system, dynamic system, user

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
        self, model_provider, prompt_config, input_with_project
    ):
        """Test static prompt handles missing GitLab context gracefully."""
        template = ChatAgentPromptTemplate(model_provider, prompt_config)

        # Call template without GitLab context
        result = template.invoke(
            input_with_project,
            agent_name="test_agent",
            is_anthropic_model=False,
        )

        messages = result.messages
        assert (
            len(messages) == 4
        )  # static system, security system, dynamic system, user

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
async def test_agentic_fake_model_bypasses_tool_approval(input, mock_toolset):
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
        toolset=mock_toolset,
        optimizer_pipeline=_make_passthrough_pipeline(),
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


@pytest.mark.asyncio
async def test_mixed_tool_calls_approval_only_for_requiring_tools(input, mock_toolset):
    """Test that approval messages are only added for tools that actually require approval.

    This test verifies the fix for the bug where approval_required flag was checked outside the loop, causing approval
    messages to be added for all tools after the first tool requiring approval.
    """
    mock_model = Mock()
    mock_model._is_auto_approved_by_agentic_mock_model = False

    mock_prompt_adapter = Mock()
    mock_prompt_adapter.get_model.return_value = mock_model

    mock_tools_registry = Mock(spec=ToolsRegistry)

    # Configure approval_required to return different values for different tools
    def approval_side_effect(tool_name, _tool_args=None):
        # preapproved_tool: no approval needed
        # tool_requiring_approval: approval needed
        # another_preapproved_tool: no approval needed
        return tool_name == "tool_requiring_approval"

    mock_tools_registry.approval_required.side_effect = approval_side_effect

    chat_agent = ChatAgent(
        name="Chat Agent",
        prompt_adapter=mock_prompt_adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
        toolset=mock_toolset,
        optimizer_pipeline=_make_passthrough_pipeline(),
    )

    # Create an AI message with multiple tool calls: preapproved, requiring approval, preapproved
    ai_message_with_mixed_tools = AIMessage(
        content="I need to use multiple tools",
        tool_calls=[
            {
                "name": "preapproved_tool",
                "args": {"param": "value1"},
                "id": "call_1",
                "type": "tool_call",
            },
            {
                "name": "tool_requiring_approval",
                "args": {"param": "value2"},
                "id": "call_2",
                "type": "tool_call",
            },
            {
                "name": "another_preapproved_tool",
                "args": {"param": "value3"},
                "id": "call_3",
                "type": "tool_call",
            },
        ],
    )

    chat_agent.prompt_adapter.get_response = AsyncMock(
        return_value=ai_message_with_mixed_tools
    )

    result = await chat_agent.run(input)

    # Should require approval because one tool needs it
    assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED

    # Should have exactly ONE approval message (only for tool_requiring_approval)
    approval_messages = [
        msg
        for msg in result["ui_chat_log"]
        if msg["message_type"] == MessageTypeEnum.REQUEST
    ]
    assert len(approval_messages) == 1
    assert approval_messages[0]["tool_info"]["name"] == "tool_requiring_approval"
    assert approval_messages[0]["tool_info"]["args"] == {"param": "value2"}


@pytest.mark.asyncio
async def test_approval_enriches_tool_info_with_project_name(input):
    """Test that approval messages use resolve_project_name_for_tool to enrich tool_info args."""
    mock_model = Mock()
    mock_model._is_auto_approved_by_agentic_mock_model = False

    mock_prompt_adapter = Mock()
    mock_prompt_adapter.get_model.return_value = mock_model

    mock_tools_registry = Mock(spec=ToolsRegistry)
    mock_tools_registry.approval_required.return_value = True

    chat_agent = ChatAgent(
        name="Chat Agent",
        prompt_adapter=mock_prompt_adapter,
        tools_registry=mock_tools_registry,
        toolset=Mock(spec=Toolset),
        system_template_override=None,
        optimizer_pipeline=_make_passthrough_pipeline(),
    )

    tool_args = {"project_id": 42, "branch": "main"}
    ai_message = AIMessage(
        content="I need to create a commit",
        tool_calls=[
            {
                "name": "create_commit",
                "args": tool_args,
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    chat_agent.prompt_adapter.get_response = AsyncMock(return_value=ai_message)

    project = Project(
        id=42,
        name="my-project",
        description="",
        http_url_to_repo="",
        web_url="",
        default_branch="main",
        languages=[],
        exclusion_rules=[],
    )
    state = {**input, "project": project}

    with patch(
        "duo_workflow_service.agents.chat_agent.resolve_project_name_for_tool",
        return_value="my-project",
    ) as mock_resolve:
        result = await chat_agent.run(state)

        mock_resolve.assert_called_once()

    approval_messages = [
        msg
        for msg in result["ui_chat_log"]
        if msg["message_type"] == MessageTypeEnum.REQUEST
    ]
    assert len(approval_messages) == 1
    assert approval_messages[0]["tool_info"]["args"]["project_name"] == "my-project"


@pytest.mark.asyncio
async def test_approval_includes_suggested_patterns_for_commands(input, mock_toolset):
    """Test that run_command tool calls with 3+ word commands include suggested_patterns."""
    mock_model = Mock()
    mock_model._is_auto_approved_by_agentic_mock_model = False

    mock_prompt_adapter = Mock()
    mock_prompt_adapter.get_model.return_value = mock_model

    mock_tools_registry = Mock(spec=ToolsRegistry)
    mock_tools_registry.approval_required = AsyncMock(return_value=True)

    chat_agent = ChatAgent(
        name="Chat Agent",
        prompt_adapter=mock_prompt_adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
        toolset=mock_toolset,
        optimizer_pipeline=_make_passthrough_pipeline(),
    )

    ai_message = AIMessage(
        content="Running a command",
        tool_calls=[
            {
                "name": "run_command",
                "args": {"command": "git checkout feature/branch"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    chat_agent.prompt_adapter.get_response = AsyncMock(return_value=ai_message)

    result = await chat_agent.run(input)

    approval_messages = [
        msg
        for msg in result["ui_chat_log"]
        if msg["message_type"] == MessageTypeEnum.REQUEST
    ]
    assert len(approval_messages) == 1
    assert approval_messages[0]["tool_info"]["suggested_patterns"] == ["git checkout *"]


@pytest.mark.asyncio
async def test_chat_agent_notifiable_exception_handling(chat_agent, input):
    """Test that ChatAgent raises NotifiableException for LLM errors."""
    error_message = "LLM service temporarily unavailable"
    original_error = APIStatusError(
        message=error_message,
        response=Mock(status_code=500),
        body={"error": {"message": "Internal server error"}},
    )

    chat_agent.prompt_adapter.get_response = AsyncMock(side_effect=original_error)

    with pytest.raises(NotifiableException) as exc_info:
        await chat_agent.run(input)

    assert str(exc_info.value) == (
        "There was an error connecting to the chosen LLM provider, please contact support "
        "if the issue persists."
    )
    assert exc_info.value.__cause__ is original_error


@pytest.mark.asyncio
async def test_chat_agent_notifiable_exception_non_5xx_error(chat_agent, input):
    """Test that ChatAgent raises NotifiableException for non-500 errors with correct message."""
    error_message = "Bad request"
    original_error = APIStatusError(
        message=error_message,
        response=Mock(status_code=400),
        body={"error": {"message": "Bad request"}},
    )

    chat_agent.prompt_adapter.get_response = AsyncMock(side_effect=original_error)

    with pytest.raises(NotifiableException) as exc_info:
        await chat_agent.run(input)

    assert str(exc_info.value) == (
        "There was an error processing your request in the Duo Agent Platform, please contact support "
        "if the issue persists."
    )
    assert exc_info.value.__cause__ is original_error


class TestAgentRetryWithPendingToolCalls:
    """Test agent behavior when resuming/retrying with pending tool_calls."""

    @pytest.mark.asyncio
    async def test_agent_inserts_synthetic_tool_messages_on_pending_calls(self):
        """Test agent inserts synthetic ToolMessages when called with pending tool_calls."""
        mock_model = Mock()
        mock_model._is_auto_approved_by_agentic_mock_model = True

        mock_prompt_adapter = Mock()
        mock_prompt_adapter.get_model.return_value = mock_model

        mock_tools_registry = Mock(spec=ToolsRegistry)
        mock_toolset = Mock(spec=Toolset)
        mock_toolset.validate_tool_call.return_value = None

        chat_agent = ChatAgent(
            name="test_agent",
            prompt_adapter=mock_prompt_adapter,
            tools_registry=mock_tools_registry,
            system_template_override=None,
            toolset=mock_toolset,
            optimizer_pipeline=_make_passthrough_pipeline(),
        )

        state = {
            "conversation_history": {
                "test_agent": [
                    HumanMessage(content="create a file"),
                    AIMessage(
                        content="I'll create the file.",
                        tool_calls=[
                            {"id": "call_1", "name": "create_file", "args": {}},
                            {"id": "call_2", "name": "write_content", "args": {}},
                        ],
                    ),
                ]
            },
            "approval": None,
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "project": None,
            "namespace": None,
        }

        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(
                content="The tool execution was interrupted. How would you like to proceed?",
                id="response-msg",
            )
        )

        result = await chat_agent.run(state)

        # Synthetic ToolMessages come back inside an Overwrite channel update;
        # the input state stays untouched (gitlab-org/gitlab#623342).
        history_update = result["conversation_history"]
        assert isinstance(history_update, Overwrite)
        history = history_update.value["test_agent"]
        assert len(history) == 5
        assert isinstance(history[2], ToolMessage)
        assert isinstance(history[3], ToolMessage)
        assert "interrupted" in history[2].content
        assert history[2].tool_call_id == "call_1"
        assert history[3].tool_call_id == "call_2"
        assert history[4].id == "response-msg"
        assert len(state["conversation_history"]["test_agent"]) == 2

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED

    @pytest.mark.asyncio
    async def test_agent_skips_llm_call_when_last_message_is_ai_without_tool_calls(
        self,
    ):
        """Test agent returns INPUT_REQUIRED when last message is AIMessage without tool_calls."""
        mock_prompt_adapter = Mock()
        mock_tools_registry = Mock(spec=ToolsRegistry)
        mock_toolset = Mock(spec=Toolset)
        mock_toolset.validate_tool_call.return_value = None

        chat_agent = ChatAgent(
            name="test_agent",
            prompt_adapter=mock_prompt_adapter,
            tools_registry=mock_tools_registry,
            system_template_override=None,
            toolset=mock_toolset,
            optimizer_pipeline=_make_passthrough_pipeline(),
        )

        state = {
            "conversation_history": {
                "test_agent": [
                    HumanMessage(content="hello"),
                    AIMessage(content="Hello! How can I help you?"),
                ]
            },
            "approval": None,
            "plan": {"steps": []},
            "status": WorkflowStatusEnum.EXECUTION,
            "ui_chat_log": [],
            "last_human_input": None,
            "project": None,
            "namespace": None,
        }

        result = await chat_agent.run(state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert result["ui_chat_log"] == []
        mock_prompt_adapter.get_response.assert_not_called()


def _successful_compaction_result() -> CompactionResult:
    summary = AIMessage(content="Summary text", id="summary-msg-id")
    return CompactionResult(
        messages=[summary, HumanMessage(content="recent")],
        was_modified=True,
        messages_summarized=3,
        compaction_input_tokens=900,
        compaction_output_tokens=150,
        summary=summary,
    )


def _make_optimizer_pipeline_with_result(
    result: OptimizationResult,
    optimized_history=None,
) -> HistoryOptimizerPipeline:
    """Pipeline mock that returns a single result and (optionally) a specific history."""
    mock_pipeline = Mock(spec=HistoryOptimizerPipeline)

    async def optimize(history):
        return (optimized_history if optimized_history is not None else history), [
            result
        ]

    mock_pipeline.optimize = AsyncMock(side_effect=optimize)
    return mock_pipeline


class TestChatAgentOptimizerPipeline:
    """ChatAgent uses ``_optimizer_pipeline`` for automatic history optimization."""

    @staticmethod
    def _build_chat_agent(
        system_template_override,
        mock_toolset,
        optimizer_pipeline,
        response_msg=None,
    ):
        mock_prompt_adapter = Mock()
        mock_prompt_adapter.get_response = AsyncMock(
            return_value=response_msg
            or AIMessage(content="Assistant reply", id="assistant-id")
        )
        mock_prompt_adapter.get_model.return_value = Mock()
        return ChatAgent(
            name="Chat Agent",
            prompt_adapter=mock_prompt_adapter,
            tools_registry=Mock(spec=ToolsRegistry),
            system_template_override=system_template_override,
            toolset=mock_toolset,
            optimizer_pipeline=optimizer_pipeline,
        )

    @staticmethod
    def _input_state():
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
    async def test_pipeline_receives_history_and_replaces_it(
        self, system_template_override, mock_toolset
    ):
        original = [HumanMessage(content="hi")]
        replaced = [HumanMessage(content="optimized")]
        result = OptimizationResult(messages=replaced, was_modified=True)
        pipeline = _make_optimizer_pipeline_with_result(
            result, optimized_history=replaced
        )
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline
        )
        state = {
            **self._input_state(),
            "conversation_history": {"Chat Agent": original},
        }

        await chat_agent.run(state)

        pipeline.optimize.assert_awaited_once_with(original)
        called_state = chat_agent.prompt_adapter.get_response.call_args[0][0]
        assert called_state["conversation_history"]["Chat Agent"] == replaced

    @pytest.mark.asyncio
    async def test_rewritten_history_returned_as_overwrite_without_mutating_state(
        self, system_template_override, mock_toolset
    ):
        """A history rewrite must come back as an Overwrite channel update.

        Mutating the input state instead would alias the checkpointer's delta baseline and mask the rewrite from
        persistence (gitlab-org/gitlab#623342).
        """
        original = [HumanMessage(content="old-1"), HumanMessage(content="old-2")]
        replaced = [AIMessage(content="summary"), HumanMessage(content="old-2")]
        result = OptimizationResult(messages=replaced, was_modified=True)
        pipeline = _make_optimizer_pipeline_with_result(
            result, optimized_history=replaced
        )
        response_msg = AIMessage(content="Assistant reply", id="assistant-id")
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline, response_msg=response_msg
        )
        other_agent_history = [AIMessage(content="other agent message")]
        state = {
            **self._input_state(),
            "conversation_history": {
                "Chat Agent": original,
                "Other Agent": other_agent_history,
            },
        }
        input_history_dict = state["conversation_history"]

        response = await chat_agent.run(state)

        history_update = response["conversation_history"]
        assert isinstance(history_update, Overwrite)
        assert history_update.value["Chat Agent"] == [*replaced, response_msg]
        assert history_update.value["Other Agent"] == other_agent_history
        assert input_history_dict["Chat Agent"] == original

    @pytest.mark.asyncio
    async def test_rewrite_reaches_checkpointer_as_compaction_delta(
        self, system_template_override, mock_toolset
    ):
        """Regression for https://gitlab.com/gitlab-org/gitlab/-/issues/623342.

        Drives the rewrite through the real conversation_history reducer channel and the checkpointer's delta
        serializer, with the baseline aliased to the live channel value the way GitLabWorkflow.aput caches it. The old
        in-place mutation made the rewrite invisible here (a plain append blob); the Overwrite update must yield a
        compaction snapshot.
        """
        original = [HumanMessage(content=f"m{i}") for i in range(6)]
        replaced = [AIMessage(content="summary"), HumanMessage(content="m5")]
        result = OptimizationResult(messages=replaced, was_modified=True)
        pipeline = _make_optimizer_pipeline_with_result(
            result, optimized_history=replaced
        )
        response_msg = AIMessage(content="Assistant reply", id="assistant-id")
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline, response_msg=response_msg
        )

        channel = BinaryOperatorAggregate(dict, _conversation_history_reducer)
        channel.update([{"Chat Agent": original}])
        live_value = channel.get()
        # GitLabWorkflow.aput caches its delta baseline as a shallow copy of
        # channel_values, so the inner dict aliases the live channel value.
        aliased_baseline = {"conversation_history": live_value}

        state = {**self._input_state(), "conversation_history": live_value}
        response = await chat_agent.run(state)
        channel.update([response["conversation_history"]])

        checkpoint = {"channel_values": {"conversation_history": channel.checkpoint()}}
        blobs, is_compaction = _serialize_channel_blobs(
            checkpoint, {"conversation_history": "2"}, aliased_baseline
        )

        assert is_compaction
        assert blobs[0]["step_action"] == "compaction"

    @pytest.mark.asyncio
    async def test_retry_correction_trace_persists_as_overwrite(
        self, system_template_override, mock_toolset
    ):
        """Malformed-tool-call retries extend the working history; the trace must persist so the stored history matches
        what the model saw."""
        malformed = AIMessage(
            content="I'll call a tool",
            tool_calls=[{"name": "read_file", "args": {"bad": True}, "id": "call_1"}],
        )
        corrected = AIMessage(content="Done without tools", id="corrected-id")
        pipeline = _make_passthrough_pipeline()
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            side_effect=[malformed, corrected]
        )
        mock_toolset.validate_tool_call.side_effect = [
            MalformedToolCallError("bad args", tool_call=malformed.tool_calls[0]),
        ]
        original = [HumanMessage(content="hi")]
        state = {
            **self._input_state(),
            "conversation_history": {"Chat Agent": original},
        }

        response = await chat_agent.run(state)

        history_update = response["conversation_history"]
        assert isinstance(history_update, Overwrite)
        persisted = history_update.value["Chat Agent"]
        assert persisted[0] == original[0]
        assert persisted[1] == malformed
        assert isinstance(persisted[2], ToolMessage)
        assert persisted[-1] == corrected
        assert state["conversation_history"]["Chat Agent"] == original

    @pytest.mark.asyncio
    async def test_unmodified_history_returned_as_plain_append(
        self, system_template_override, mock_toolset
    ):
        result = OptimizationResult(
            messages=[HumanMessage(content="hi")], was_modified=False
        )
        pipeline = _make_optimizer_pipeline_with_result(result)
        response_msg = AIMessage(content="Assistant reply", id="assistant-id")
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline, response_msg=response_msg
        )

        response = await chat_agent.run(self._input_state())

        assert response["conversation_history"] == {"Chat Agent": [response_msg]}

    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.chat_agent.log_exception")
    async def test_rewrite_survives_slash_command_validation_error(
        self, _mock_log_exception, system_template_override, mock_toolset
    ):
        original = [HumanMessage(content="old")]
        replaced = [AIMessage(content="summary")]
        result = OptimizationResult(messages=replaced, was_modified=True)
        pipeline = _make_optimizer_pipeline_with_result(
            result, optimized_history=replaced
        )
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            side_effect=SlashCommandValidationError("The command does not exist.")
        )
        state = {
            **self._input_state(),
            "conversation_history": {"Chat Agent": original},
        }

        response = await chat_agent.run(state)

        history_update = response["conversation_history"]
        assert isinstance(history_update, Overwrite)
        assert history_update.value["Chat Agent"][:-1] == replaced
        assert isinstance(history_update.value["Chat Agent"][-1], AIMessage)
        assert state["conversation_history"]["Chat Agent"] == original

    @pytest.mark.asyncio
    async def test_auto_compaction_emits_tool_card_after_assistant_entry(
        self, system_template_override, mock_toolset
    ):
        """When the pipeline surfaces a compaction UI entry, it is appended after the assistant entry."""
        result = _successful_compaction_result()
        # Simulate the CompactionOptimizer populating ui_chat_logs on the result.
        result.ui_chat_logs = [
            build_compaction_tool_card(
                trigger="auto",
                result=result,
                status=ToolStatus.SUCCESS,
            )
        ]
        pipeline = _make_optimizer_pipeline_with_result(result)
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline
        )

        response = await chat_agent.run(self._input_state())

        ui_logs = response["ui_chat_log"]
        assert len(ui_logs) == 2
        assistant_entry, compaction_entry = ui_logs
        assert assistant_entry["message_type"] == MessageTypeEnum.AGENT
        assert assistant_entry["content"] == "Assistant reply"
        assert compaction_entry["message_type"] == MessageTypeEnum.TOOL
        assert compaction_entry["message_sub_type"] == "compaction"
        assert compaction_entry["tool_info"]["args"]["trigger"] == "auto"
        assert compaction_entry["tool_info"]["args"]["messages_summarized"] == 3

    @pytest.mark.asyncio
    async def test_no_ui_logs_when_optimizer_produces_none(
        self, system_template_override, mock_toolset
    ):
        result = OptimizationResult(messages=[HumanMessage(content="hi")])
        pipeline = _make_optimizer_pipeline_with_result(result)
        chat_agent = self._build_chat_agent(
            system_template_override, mock_toolset, pipeline
        )

        response = await chat_agent.run(self._input_state())

        assert len(response["ui_chat_log"]) == 1
        assert response["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT

    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.chat_agent.log_exception")
    async def test_optimizer_ui_logs_survive_slash_command_validation_error(
        self,
        _mock_log_exception,
        system_template_override,
        mock_toolset,
    ):
        """If the LLM call raises SlashCommandValidationError after optimization ran, the compaction entry stays."""
        result = _successful_compaction_result()
        result.ui_chat_logs = [
            build_compaction_tool_card(
                trigger="auto",
                result=result,
                status=ToolStatus.SUCCESS,
            )
        ]
        pipeline = _make_optimizer_pipeline_with_result(result)

        mock_prompt_adapter = Mock()
        mock_prompt_adapter.get_response = AsyncMock(
            side_effect=SlashCommandValidationError(
                "The command '/invalid' does not exist."
            )
        )
        mock_prompt_adapter.get_model.return_value = Mock()

        chat_agent = ChatAgent(
            name="Chat Agent",
            prompt_adapter=mock_prompt_adapter,
            tools_registry=Mock(spec=ToolsRegistry),
            system_template_override=system_template_override,
            toolset=mock_toolset,
            optimizer_pipeline=pipeline,
        )

        response = await chat_agent.run(self._input_state())

        ui_logs = response["ui_chat_log"]
        assert len(ui_logs) == 2
        assert ui_logs[0]["message_id"].startswith("error-")
        assert ui_logs[1]["message_sub_type"] == "compaction"


def _manual_success_result() -> CompactionResult:
    summary = AIMessage(content="Summary text", id="summary-msg-id")
    result = CompactionResult(
        messages=[summary, HumanMessage(content="recent")],
        was_modified=True,
        messages_summarized=3,
        compaction_input_tokens=900,
        compaction_output_tokens=150,
        summary=summary,
    )
    # Mirror what CompactionOptimizer.optimize_manual populates.
    result.ui_chat_logs = [
        build_compaction_tool_card(
            trigger="manual",
            result=result,
            status=ToolStatus.SUCCESS,
        )
    ]
    return result


def _manual_failure_result(error: Exception | None = None) -> CompactionResult:
    result = CompactionResult(
        messages=[HumanMessage(content="orig")],
        was_modified=False,
        error=error,
    )
    result.ui_chat_logs = [
        build_compaction_tool_card(
            trigger="manual",
            result=result,
            content="Compaction failed",
            status=ToolStatus.FAILURE,
        ),
    ]
    return result


class TestChatAgentManualCompaction:
    """/compact slash command dispatches to ``_manual_compactor.optimize_manual``."""

    @pytest.fixture(name="mock_manual_compactor")
    def mock_manual_compactor_fixture(self):
        mock = Mock()
        mock.optimize_manual = AsyncMock(
            return_value=_manual_success_result(),
        )
        return mock

    @pytest.fixture(name="chat_agent")
    def chat_agent_fixture(
        self, system_template_override, mock_toolset, mock_manual_compactor
    ):
        mock_prompt_adapter = Mock()
        mock_prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(content="LLM response", id="llm-msg-id")
        )
        mock_prompt_adapter.get_model.return_value = Mock()
        return ChatAgent(
            name="Chat Agent",
            prompt_adapter=mock_prompt_adapter,
            tools_registry=Mock(spec=ToolsRegistry),
            system_template_override=system_template_override,
            toolset=mock_toolset,
            optimizer_pipeline=_make_passthrough_pipeline(),
            manual_compactor=mock_manual_compactor,
        )

    @staticmethod
    def _state_with_history(input_state, history):
        return {**input_state, "conversation_history": {"Chat Agent": list(history)}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "last_user_message, expected_compaction_called, expected_llm_called",
        [
            pytest.param("/compact", True, False, id="compact_no_args"),
            pytest.param(
                "/compact focus on the auth bug",
                True,
                False,
                id="compact_with_trailing_text",
            ),
            pytest.param("  /compact  ", True, False, id="compact_with_whitespace"),
            pytest.param("/explain this code", False, True, id="explain_falls_through"),
            pytest.param("hello", False, True, id="plain_text_falls_through"),
            pytest.param(
                "/home/user/file.py", False, True, id="file_path_not_a_command"
            ),
        ],
    )
    async def test_routes_compact_slash_command(
        self,
        last_user_message,
        expected_compaction_called,
        expected_llm_called,
        chat_agent,
        mock_manual_compactor,
        input,
    ):
        prior_history = [
            HumanMessage(content="earlier"),
            AIMessage(content="reply"),
        ]
        full_history = prior_history + [HumanMessage(content=last_user_message)]
        state = self._state_with_history(input, full_history)

        await chat_agent.run(state)

        assert (
            mock_manual_compactor.optimize_manual.called is expected_compaction_called
        )
        assert chat_agent.prompt_adapter.get_response.called is expected_llm_called

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "last_user_message, expected_user_instruction",
        [
            pytest.param("/compact", None, id="no_trailing_text"),
            pytest.param(
                "/compact focus on auth bug",
                "focus on auth bug",
                id="with_trailing_text",
            ),
        ],
    )
    async def test_success_path_invocation_and_outcome(
        self,
        last_user_message,
        expected_user_instruction,
        chat_agent,
        mock_manual_compactor,
        input,
    ):
        """On success the /compact command is stripped, user_instruction is forwarded, the UI log carries the summary
        tool card produced by the CompactionOptimizer, and conversation_history is returned as an Overwrite update."""
        prior_history = [
            HumanMessage(content="task"),
            AIMessage(content="working on it"),
        ]
        full_history = prior_history + [HumanMessage(content=last_user_message)]
        state = self._state_with_history(input, full_history)
        compactor_result = mock_manual_compactor.optimize_manual.return_value

        result = await chat_agent.run(state)

        # The /compact message is stripped before being passed to the compactor.
        assert mock_manual_compactor.optimize_manual.call_args.args[0] == prior_history
        assert (
            mock_manual_compactor.optimize_manual.call_args.kwargs["user_instruction"]
            == expected_user_instruction
        )

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert result["ui_chat_log"] == list(compactor_result.ui_chat_logs)
        # The compacted history comes back as a channel Overwrite; the input
        # state must stay untouched because it aliases the checkpointer's
        # delta baseline (gitlab-org/gitlab#623342).
        history_update = result["conversation_history"]
        assert isinstance(history_update, Overwrite)
        assert history_update.value["Chat Agent"] == compactor_result.messages
        assert state["conversation_history"]["Chat Agent"] == full_history

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "compactor_result, expected_error_type",
        [
            pytest.param(
                _manual_failure_result(error=RuntimeError("boom")),
                "RuntimeError",
                id="llm_error",
            ),
            pytest.param(
                _manual_failure_result(),
                None,
                id="no_summary_field",
            ),
        ],
    )
    async def test_failure_leaves_history_unchanged(
        self,
        compactor_result,
        expected_error_type,
        chat_agent,
        mock_manual_compactor,
        input,
    ):
        mock_manual_compactor.optimize_manual.return_value = compactor_result
        full_history = [
            HumanMessage(content="initial"),
            AIMessage(content="reply"),
            HumanMessage(content="/compact"),
        ]
        state = self._state_with_history(input, full_history)

        with patch("duo_workflow_service.agents.chat_agent.log") as mock_log:
            result = await chat_agent.run(state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert result["ui_chat_log"] == list(compactor_result.ui_chat_logs)
        # History is unchanged.
        assert state["conversation_history"]["Chat Agent"] == full_history
        mock_log.warning.assert_called_once()
        assert mock_log.warning.call_args.kwargs["error_type"] == expected_error_type

    @pytest.mark.asyncio
    async def test_no_prior_history_returns_friendly_notice(
        self, chat_agent, mock_manual_compactor, input
    ):
        """When /compact is the only message, return a friendly notice without invoking the compactor."""
        state = self._state_with_history(input, [HumanMessage(content="/compact")])

        result = await chat_agent.run(state)

        mock_manual_compactor.optimize_manual.assert_not_called()
        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert len(result["ui_chat_log"]) == 2
        tool_entry, agent_entry = result["ui_chat_log"]
        assert tool_entry["message_type"] == MessageTypeEnum.TOOL
        assert tool_entry["message_sub_type"] == "compaction"
        assert tool_entry["status"] == ToolStatus.SUCCESS
        assert tool_entry["content"] == "Nothing to compact"
        assert "tool_response" not in tool_entry["tool_info"]
        assert agent_entry["message_type"] == MessageTypeEnum.AGENT
        assert "no conversation history" in agent_entry["content"].lower()

    @pytest.mark.asyncio
    async def test_manual_compactor_is_none_returns_error(
        self, system_template_override, mock_toolset, input
    ):
        chat_agent = ChatAgent(
            name="Chat Agent",
            prompt_adapter=Mock(),
            tools_registry=Mock(spec=ToolsRegistry),
            system_template_override=system_template_override,
            toolset=mock_toolset,
            optimizer_pipeline=_make_passthrough_pipeline(),
            manual_compactor=None,
        )
        state = TestChatAgentManualCompaction._state_with_history(
            input, [HumanMessage(content="/compact")]
        )

        result = await chat_agent.run(state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert len(result["ui_chat_log"]) == 2
        tool_entry, agent_entry = result["ui_chat_log"]
        assert tool_entry["message_type"] == MessageTypeEnum.TOOL
        assert tool_entry["status"] == ToolStatus.FAILURE
        assert tool_entry["content"] == "Compaction failed"
        assert "tool_response" not in tool_entry["tool_info"]
        assert agent_entry["message_type"] == MessageTypeEnum.AGENT
        assert "not available" in agent_entry["content"].lower()


class TestSuggestPatterns:
    """Tests for _suggest_patterns helper."""

    def test_command_tool_shell_form(self):
        patterns = _suggest_patterns(
            "run_command", {"command": "git checkout feature/branch"}
        )
        assert patterns == ["git checkout *"]

    def test_command_tool_split_args_form(self):
        patterns = _suggest_patterns(
            "run_command", {"program": "npm", "args": "install lodash"}
        )
        assert patterns == ["npm install *"]

    def test_command_tool_single_word(self):
        patterns = _suggest_patterns("run_command", {"command": "ls"})
        assert not patterns

    def test_command_tool_two_words(self):
        patterns = _suggest_patterns("run_command", {"command": "git status"})
        assert not patterns

    def test_non_command_tool(self):
        patterns = _suggest_patterns("read_file", {"path": "/tmp/test.txt"})
        assert not patterns

    def test_empty_command(self):
        patterns = _suggest_patterns("run_command", {"command": ""})
        assert not patterns

    def test_program_only_no_args(self):
        patterns = _suggest_patterns("run_command", {"program": "ls"})
        assert not patterns

    def test_long_command(self):
        patterns = _suggest_patterns(
            "run_command", {"command": "docker compose -f dev.yml up -d"}
        )
        assert patterns == [
            "docker compose -f dev.yml up *",
            "docker compose *",
        ]

    def test_git_command_tool(self):
        patterns = _suggest_patterns(
            "run_git_command",
            {
                "command": "checkout",
                "args": "feature/my-branch",
                "repository_url": "https://example.com/repo.git",
            },
        )
        assert patterns == ["git checkout *"]

    def test_git_command_tool_long_args(self):
        patterns = _suggest_patterns(
            "run_git_command",
            {
                "command": "push",
                "args": "origin feature/my-branch",
                "repository_url": "https://example.com/repo.git",
            },
        )
        assert patterns == ["git push origin *", "git push *"]

    def test_git_command_tool_no_args(self):
        patterns = _suggest_patterns(
            "run_git_command",
            {"command": "status", "repository_url": "https://example.com/repo.git"},
        )
        assert not patterns

    def test_git_command_tool_simple_args(self):
        patterns = _suggest_patterns(
            "run_git_command",
            {
                "command": "add",
                "args": ".",
                "repository_url": "https://example.com/repo.git",
            },
        )
        assert patterns == ["git add *"]


def _web_search_ai_message(msg_id="agent-msg-id"):
    """A final AIMessage carrying Anthropic server-side web-search content blocks."""
    return AIMessage(
        content=[
            {"type": "text", "text": "Let me look that up."},
            {
                "type": "server_tool_use",
                "id": "srvtu_1",
                "name": "web_search",
                "input": {"query": "gitlab duo"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtu_1",
                "content": [{"type": "web_search_result", "url": "https://x"}],
            },
            {"type": "text", "text": " Here is what I found."},
        ],
        id=msg_id,
    )


class TestServerToolResponse:
    """Agent-response-path projection of server_tool_use blocks."""

    def test_splits_text_around_tool_with_success_card(self, chat_agent):
        result = {}

        chat_agent._build_ui_chat_log(
            _web_search_ai_message(), {"conversation_history": {}}, result
        )

        log = result["ui_chat_log"]
        assert [e["message_type"] for e in log] == [
            MessageTypeEnum.AGENT,
            MessageTypeEnum.TOOL,
            MessageTypeEnum.AGENT,
        ]
        pre, tool, summary = log
        assert pre["message_id"] == "agent-msg-id"
        assert pre["content"] == "Let me look that up."
        assert tool["message_sub_type"] == "web_search"
        assert tool["status"] == ToolStatus.SUCCESS
        assert tool["message_id"] == "srvtu_1"
        assert tool["tool_info"]["args"] == {"query": "gitlab duo"}
        assert tool["tool_info"]["tool_response"] == [
            {"type": "web_search_result", "url": "https://x"}
        ]
        assert summary["message_id"] == "agent-msg-id:seg1"
        assert summary["content"] == " Here is what I found."

    def test_pending_card_when_result_block_absent(self, chat_agent):
        msg = AIMessage(
            content=[
                {
                    "type": "server_tool_use",
                    "id": "srvtu_1",
                    "name": "web_search",
                    "input": {},
                },
            ],
            id="agent-msg-id",
        )
        result = {}

        chat_agent._build_ui_chat_log(msg, {"conversation_history": {}}, result)

        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["status"] == ToolStatus.PENDING

    def test_plain_string_produces_single_agent_entry(self, chat_agent):
        result = {}
        chat_agent._build_ui_chat_log(
            AIMessage(content="plain text", id="msg-1"),
            {"conversation_history": {}},
            result,
        )
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["content"] == "plain text"

    def test_list_content_with_only_text_blocks_produces_single_agent_entry(
        self, chat_agent
    ):
        msg = AIMessage(
            content=[{"type": "text", "text": "Hello world"}],
            id="msg-1",
        )
        result = {}
        chat_agent._build_ui_chat_log(msg, {"conversation_history": {}}, result)
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["content"] == "Hello world"

    def test_client_tool_use_block_produces_no_server_card(self, chat_agent):
        msg = AIMessage(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "read_file",
                    "input": {"path": "x"},
                },
            ],
            id="agent-msg-id",
        )
        result = {}
        chat_agent._build_ui_chat_log(msg, {"conversation_history": {}}, result)
        tool_entries = [
            e
            for e in result["ui_chat_log"]
            if e["message_type"] == MessageTypeEnum.TOOL
        ]
        assert tool_entries == []


class TestToolApprovalRequestTracking:
    """Test suite for approval-request internal event tracking in ChatAgent."""

    @staticmethod
    def _make_agent(mock_toolset, internal_event_client, approval_side_effect):
        mock_model = Mock()
        mock_model._is_auto_approved_by_agentic_mock_model = False

        mock_prompt_adapter = Mock()
        mock_prompt_adapter.get_model.return_value = mock_model

        mock_tools_registry = Mock(spec=ToolsRegistry)
        mock_tools_registry.approval_required.side_effect = approval_side_effect

        return ChatAgent(
            name="Chat Agent",
            prompt_adapter=mock_prompt_adapter,
            tools_registry=mock_tools_registry,
            system_template_override=None,
            toolset=mock_toolset,
            optimizer_pipeline=_make_passthrough_pipeline(),
            tracker=ToolEventTracker(
                flow_id="wf-123",
                flow_type=GLReportingEventContext.from_workflow_definition("chat"),
                internal_event_client=internal_event_client,
            ),
        )

    @pytest.mark.asyncio
    async def test_tool_needing_approval_tracks_request_event(
        self, input, mock_toolset, internal_event_client
    ):
        """A tool call requiring approval tracks one request event with no tool args."""
        chat_agent = self._make_agent(
            mock_toolset, internal_event_client, lambda *_args, **_kwargs: True
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(
                content="I need to use a tool",
                tool_calls=[
                    {
                        "name": "run_command",
                        "args": {"command": "rm -rf /tmp/secret-path"},
                        "id": "call_123",
                        "type": "tool_call",
                    }
                ],
            )
        )

        result = await chat_agent.run(input)

        assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
        internal_event_client.track_event.assert_called_once_with(
            event_name="request_duo_workflow_tool_approval",
            additional_properties=InternalEventAdditionalProperties(
                label="chat",
                property="run_command",
                value="wf-123",
                tool_name="run_command",
            ),
            category="chat",
        )
        # Privacy: command text must never appear in the payload
        assert "rm -rf" not in str(internal_event_client.track_event.call_args.kwargs)

    @pytest.mark.asyncio
    async def test_batch_of_two_tools_needing_approval_tracks_two_request_events(
        self, input, mock_toolset, internal_event_client
    ):
        """Two tools needing approval track two request events."""
        chat_agent = self._make_agent(
            mock_toolset, internal_event_client, lambda *_args, **_kwargs: True
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(
                content="I need to use two tools",
                tool_calls=[
                    {
                        "name": "run_command",
                        "args": {"command": "ls"},
                        "id": "call_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "edit_file",
                        "args": {"path": "a.py"},
                        "id": "call_2",
                        "type": "tool_call",
                    },
                ],
            )
        )

        result = await chat_agent.run(input)

        assert result["status"] == WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
        assert internal_event_client.track_event.call_count == 2
        tracked_tools = [
            call.kwargs["additional_properties"].extra["tool_name"]
            for call in internal_event_client.track_event.call_args_list
        ]
        assert tracked_tools == ["run_command", "edit_file"]

    @pytest.mark.asyncio
    async def test_denied_tool_call_tracks_block_event(
        self, input, mock_toolset, internal_event_client
    ):
        """A response calling a denied tool tracks one block event at validation time."""
        chat_agent = self._make_agent(
            mock_toolset, internal_event_client, lambda *_args, **_kwargs: True
        )
        denied_call = {
            "name": "denied_tool",
            "args": {"command": "rm -rf /tmp/secret-path"},
            "id": "call_1",
            "type": "tool_call",
        }
        mock_toolset.denied_tools = {"denied_tool"}
        mock_toolset.validate_tool_call.side_effect = MalformedToolCallError(
            "Tool: 'denied_tool' not found", tool_call=denied_call
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(content="I need a tool", tool_calls=[denied_call])
        )

        await chat_agent.run(input)

        block_calls = [
            call
            for call in internal_event_client.track_event.call_args_list
            if call.kwargs["event_name"] == "block_denied_duo_workflow_tool"
        ]
        assert len(block_calls) == 1
        assert block_calls[0].kwargs[
            "additional_properties"
        ] == InternalEventAdditionalProperties(
            label="chat",
            property="denied_tool",
            value="wf-123",
            tool_name="denied_tool",
        )
        # Privacy: command text must never appear in the payload
        assert "rm -rf" not in str(internal_event_client.track_event.call_args_list)

    @pytest.mark.asyncio
    async def test_invalid_non_denied_tool_tracks_no_block_event(
        self, input, mock_toolset, internal_event_client
    ):
        """A malformed call to a tool no deny rule covers tracks no block event."""
        chat_agent = self._make_agent(
            mock_toolset, internal_event_client, lambda *_args, **_kwargs: True
        )
        invalid_call = {
            "name": "hallucinated_tool",
            "args": {},
            "id": "call_1",
            "type": "tool_call",
        }
        mock_toolset.denied_tools = {"some_other_denied_tool"}
        mock_toolset.validate_tool_call.side_effect = MalformedToolCallError(
            "Tool: 'hallucinated_tool' not found", tool_call=invalid_call
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(content="I need a tool", tool_calls=[invalid_call])
        )

        await chat_agent.run(input)

        internal_event_client.track_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_pre_approved_tool_tracks_no_request_event(
        self, input, mock_toolset, internal_event_client
    ):
        """A tool call not requiring approval tracks no request event."""
        chat_agent = self._make_agent(
            mock_toolset, internal_event_client, lambda *_args, **_kwargs: False
        )
        chat_agent.prompt_adapter.get_response = AsyncMock(
            return_value=AIMessage(
                content="I need to use a tool",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"path": "a.py"},
                        "id": "call_123",
                        "type": "tool_call",
                    }
                ],
            )
        )

        result = await chat_agent.run(input)

        assert result["status"] == WorkflowStatusEnum.EXECUTION
        internal_event_client.track_event.assert_not_called()
