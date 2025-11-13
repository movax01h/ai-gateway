from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_gateway.prompts.base import Prompt
from duo_workflow_service.agents.agent import Agent, AgentPromptTemplate, build_agent
from duo_workflow_service.entities import WorkflowEventType
from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    MessageTypeEnum,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="prompt_template_factory")
def promtp_template_factory_fixture():
    return AgentPromptTemplate


@pytest.fixture(name="prompt_template")
def prompt_template_fixture() -> dict[str, str]:
    return {
        "system": "You are AGI entity capable of anything",
        "user": "{% if handover %}Handover: {{handover}}.\n{% endif %}Your goal is: {{ goal }}",
    }


@pytest.fixture(name="check_events")
def check_events_fixture() -> bool:
    return True


@pytest.fixture(name="agent")
def agent_fixture(
    prompt: Prompt,
    gl_http_client: GitlabHttpClient,
    workflow_type: CategoryEnum,
    check_events: bool,
) -> Agent:
    return Agent(
        name=prompt.name,
        prompt=prompt,
        workflow_id="test-workflow-123",
        workflow_type=workflow_type,
        http_client=gl_http_client,
        check_events=check_events,
    )  # type: ignore[call-arg]


class TestAgent:
    @pytest.mark.asyncio
    async def test_run_with_empty_conversation(
        self,
        agent: Agent,
        workflow_state: DuoWorkflowStateType,
        prompt_name: str,
        goal: str,
        model_response: str,
    ):
        result = await agent.run(workflow_state)

        assert result["conversation_history"][prompt_name] == [
            SystemMessage(content="You are AGI entity capable of anything"),
            HumanMessage(content=f"Your goal is: {goal}"),
            AIMessage.model_construct(
                content=model_response, id=ANY, usage_metadata=ANY
            ),
        ]

    @pytest.mark.asyncio
    async def test_run_with_empty_conversation_and_handover(
        self,
        agent: Agent,
        workflow_state: WorkflowState,
        prompt_name: str,
        goal: str,
        model_response: str,
    ):
        workflow_state["handover"] = [
            SystemMessage(content="System message"),
            HumanMessage(content="Human message"),
        ]
        expected_handover = """================================ System Message ================================

System message
================================ Human Message =================================

Human message"""

        result = await agent.run(workflow_state)

        assert result["conversation_history"][prompt_name] == [
            SystemMessage(content="You are AGI entity capable of anything"),
            HumanMessage(
                content=f"Handover: {expected_handover}.\nYour goal is: {goal}"
            ),
            AIMessage.model_construct(
                content=model_response, id=ANY, usage_metadata=ANY
            ),
        ]

    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.agent.get_event")
    @pytest.mark.parametrize("check_events", [True, False])
    @pytest.mark.parametrize(
        "last_human_input",
        [
            None,
            {
                "id": "event-id",
                "event_type": "resume",
                "message": "",
            },
        ],
    )
    async def test_run_without_chat_log(
        self,
        mock_get_event: Mock,
        agent: Agent,
        workflow_state: DuoWorkflowStateType,
        prompt_name: str,
        model_response: str,
        check_events: bool,
    ):
        workflow_state["conversation_history"][prompt_name] = [
            HumanMessage(content="Existing chat")
        ]

        result = await agent.run(workflow_state)

        if check_events:
            mock_get_event.assert_called_once_with(
                agent.http_client, agent.workflow_id, False
            )
        else:
            mock_get_event.assert_not_called()

        # Check that we have exactly one AIMessage with the expected content
        assert len(result["conversation_history"][prompt_name]) == 1
        message = result["conversation_history"][prompt_name][0]
        assert isinstance(message, AIMessage)
        assert message.content == model_response
        # Don't check for exact equality since the message may have additional fields like id and usage_metadata

        assert "ui_chat_log" not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "last_human_input",
        [
            {
                "id": "event-id",
                "event_type": "message",
                "message": "this is a test message",
            }
        ],
    )
    @pytest.mark.parametrize(
        ("model_response", "expected_content"),
        [
            ("simple string", "simple string"),
            ([["Line 1", "Line 2", "Line 3"]], "Line 1\nLine 2\nLine 3"),
            ([[{"text": "Message from dict"}, {"other": "data"}]], "Message from dict"),
            ([[{"other": "data"}, {"more": "data"}]], None),
        ],
    )
    async def test_run_with_chat_log(
        self,
        agent: Agent,
        workflow_state: DuoWorkflowStateType,
        expected_content: str | None,
    ):
        result = await agent.run(workflow_state)

        if expected_content:
            assert len(result["ui_chat_log"]) == 1
            assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
            assert result["ui_chat_log"][0]["content"] == expected_content
        else:
            assert len(result["ui_chat_log"]) == 0

    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.agent.get_event")
    @pytest.mark.parametrize(
        ("event", "expected_status"),
        [
            (
                {
                    "id": "event-id",
                    "event_type": WorkflowEventType.STOP,
                    "message": "Workflow cancelled",
                },
                WorkflowStatusEnum.CANCELLED,
            ),
            (
                {
                    "id": "event-id",
                    "event_type": "message",
                    "message": "Continue working",
                },
                None,
            ),
        ],
    )
    async def test_run_with_cancelled_workflow(
        self,
        mock_get_event: Mock,
        agent: Agent,
        workflow_state: DuoWorkflowStateType,
        event: WorkflowEvent | None,
        expected_status: WorkflowStatusEnum,
    ):
        mock_get_event.return_value = event
        result = await agent.run(workflow_state)

        if expected_status:
            assert result["status"] == expected_status
        else:
            assert "status" not in result

    @pytest.mark.asyncio
    async def test_run_with_4xx_api_error(
        self,
        agent: Agent,
        workflow_state: DuoWorkflowStateType,
        prompt_name: str,
    ):
        # Mock the superclass ainvoke method to raise an APIStatusError
        with patch.object(
            agent.__class__.__bases__[0], "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.side_effect = APIStatusError(
                message="Test API error",
                response=Mock(status_code=400),
                body={"error": {"message": "Bad request"}},
            )

            result = await agent.run(workflow_state)

            # Verify error response structure
            assert result["status"] == WorkflowStatusEnum.ERROR
            assert "conversation_history" in result
            assert result["conversation_history"][prompt_name][0].content.startswith(
                "There was an error processing your request:"
            )
            assert len(result["ui_chat_log"]) == 1
            assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
            assert result["ui_chat_log"][0]["status"] == ToolStatus.FAILURE
            assert (
                result["ui_chat_log"][0]["content"]
                == "There was an error processing your request in the Duo Agent Platform, please try again or "
                "contact support if the issue persists."
            )

    @pytest.mark.asyncio
    async def test_run_with_5xx_api_error(
        self,
        agent: Agent,
        workflow_state: DuoWorkflowStateType,
        prompt_name: str,
    ):
        # Mock the superclass ainvoke method to raise an APIStatusError
        with patch.object(
            agent.__class__.__bases__[0], "ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke:
            mock_ainvoke.side_effect = APIStatusError(
                message="Test API error",
                response=Mock(status_code=500),
                body={"error": {"message": "Internal server error"}},
            )

            result = await agent.run(workflow_state)

            # Verify error response structure
            assert result["status"] == WorkflowStatusEnum.ERROR
            assert "conversation_history" in result
            assert result["conversation_history"][prompt_name][0].content.startswith(
                "There was an error processing your request:"
            )
            assert len(result["ui_chat_log"]) == 1
            assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
            assert result["ui_chat_log"][0]["status"] == ToolStatus.FAILURE
            assert (
                result["ui_chat_log"][0]["content"]
                == "There was an error connecting to the chosen LLM provider, please try again or contact "
                "support if the issue persists."
            )


def test_create_agent_with_prompt_registry(
    user, mock_local_prompt_registry, tools, prompt, gl_http_client
):
    agent = build_agent(
        name="test_agent",
        prompt_registry=mock_local_prompt_registry,
        user=user,
        prompt_id="test/agent",
        prompt_version="^1.0.0",
        internal_event_category="test_category",
        tools=tools,
        workflow_id="workflow_123",
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        check_events=True,
        http_client=gl_http_client,
    )

    mock_local_prompt_registry.get_on_behalf.assert_called_once_with(
        user,
        "test/agent",
        "^1.0.0",
        tools=tools,
        internal_event_extra={
            "agent_name": "test_agent",
            "workflow_id": "workflow_123",
            "workflow_type": CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        },
    )

    assert isinstance(agent, Agent)
    assert agent.name == "test_agent"
    assert agent.workflow_id == "workflow_123"
    assert agent.prompt == prompt
