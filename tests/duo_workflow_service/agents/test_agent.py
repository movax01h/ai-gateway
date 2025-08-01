# pylint: disable=unused-import,unused-variable

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
from xml.etree import ElementTree

import pytest
from anthropic import APIStatusError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from duo_workflow_service.agents import Agent
from duo_workflow_service.agents.prompts import HANDOVER_TOOL_NAME
from duo_workflow_service.entities import WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.errors.error_handler import (
    ModelError,
    ModelErrorHandler,
    ModelErrorType,
)
from duo_workflow_service.tools import Toolset
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum


@pytest.fixture(name="workflow_msg_event")
def workflow_msg_event_fixture():
    return {
        "id": "event-id",
        "event_type": "message",
        "message": "this is a test message",
    }


@pytest.fixture(name="workflow_resume_event")
def workflow_resume_event_fixture():
    return {
        "id": "event-id",
        "event_type": "resume",
        "message": "",
    }


@pytest.fixture(name="workflow_state")
def workflow_state_fixture(plan: Plan):
    return WorkflowState(
        plan=plan,
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
        project=None,
        goal=None,
        additional_context=None,
    )


# pylint: disable=too-many-public-methods
class TestAgent:
    @pytest.fixture(name="chat_mock")
    def chat_mock_fixture(self):
        mock = MagicMock(BaseChatModel)
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture(name="mock_toolset")
    def mock_toolset_fixture(self):
        mock = MagicMock(spec=Toolset)
        mock.bindable = []
        return mock

    @pytest.fixture(name="planner_agent")
    def planner_agent_fixture(
        self, chat_mock, gl_http_client, mock_toolset, internal_event_client
    ):
        return Agent(
            goal="Make the world a better place",
            model=chat_mock,
            name="test agent",
            system_prompt="You are AGI entity capable of anything",
            toolset=mock_toolset,
            workflow_id="test-workflow-123",
            http_client=gl_http_client,
            workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            internal_event_client=internal_event_client,
        )

    def test_init(
        self,
        chat_mock,
        planner_agent,
        gl_http_client,
        mock_toolset,
        internal_event_client,
    ):
        assert planner_agent._goal == "Make the world a better place"
        assert planner_agent._model == chat_mock
        assert planner_agent.name == "test agent"
        assert planner_agent._system_prompt == "You are AGI entity capable of anything"
        assert planner_agent._workflow_id == "test-workflow-123"
        assert planner_agent._http_client == gl_http_client
        assert planner_agent._toolset == mock_toolset
        assert planner_agent._internal_event_client == internal_event_client
        chat_mock.bind_tools.assert_called_once_with(mock_toolset.bindable)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_duo_workflow_service_container")
    async def test_run_with_empty_conversation(
        self, chat_mock, planner_agent, workflow_state
    ):
        chat_mock.ainvoke.return_value = AIMessage(content="test")

        result = await planner_agent.run(workflow_state)

        chat_mock.ainvoke.assert_called_once()
        assert result["conversation_history"]["test agent"] == [
            SystemMessage(content="You are AGI entity capable of anything"),
            HumanMessage(content="Your goal is: Make the world a better place"),
            AIMessage(content="test"),
        ]

    @pytest.mark.asyncio
    async def test_run_with_empty_conversation_and_handover(
        self, chat_mock, planner_agent, workflow_state
    ):
        chat_mock.ainvoke.return_value = AIMessage(content="test")
        workflow_state["handover"] = [
            HumanMessage(
                content="I tried to tell jokes on the streets to cheer everybody up"
            )
        ]

        result = await planner_agent.run(workflow_state)

        chat_mock.ainvoke.assert_called_once()
        assert result["conversation_history"]["test agent"] == [
            SystemMessage(content="You are AGI entity capable of anything"),
            HumanMessage(
                content="The steps towards goal accomplished so far are as follow:"
            ),
            HumanMessage(
                content="I tried to tell jokes on the streets to cheer everybody up"
            ),
            HumanMessage(content="Your goal is: Make the world a better place"),
            AIMessage(content="test"),
        ]

    # pylint: disable=too-many-positional-arguments
    @pytest.mark.asyncio
    async def test_run(
        self, chat_mock, planner_agent, workflow_state, internal_event_client: Mock
    ):
        workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT

        tool_calls = [
            {
                "id": "call_1",
                "name": "search_code",
                "args": {"query": "find main function"},
            },
            {
                "id": "call_2",
                "name": HANDOVER_TOOL_NAME,
                "args": {"summary": "Analyzing code structure"},
            },
        ]

        chat_mock.ainvoke.return_value = AIMessage(
            content="42",
            tool_calls=tool_calls,
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

        workflow_state["conversation_history"]["test agent"] = [
            HumanMessage(
                content="What is the answer to life the universe and everything?"
            )
        ]

        result = await planner_agent.run(workflow_state)

        chat_mock.ainvoke.assert_called_once()

        assert result["conversation_history"]["test agent"] == [
            AIMessage(
                content="42",
                tool_calls=tool_calls,
                usage_metadata={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
        ]

        assert internal_event_client.track_event.call_count == 1
        internal_event_client.track_event.assert_has_calls(
            [
                call(
                    event_name=EventEnum.TOKEN_PER_USER_PROMPT.value,
                    additional_properties=InternalEventAdditionalProperties(
                        label="test agent",
                        property=EventPropertyEnum.WORKFLOW_ID.value,
                        value="undefined",
                        input_tokens=1,
                        output_tokens=1,
                        total_tokens=2,
                        estimated_input_tokens=22,
                    ),
                    category=workflow_type,
                )
            ]
        )

    # pylint: enable=too-many-positional-arguments

    @pytest.mark.asyncio
    async def test_run_with_string_content(
        self, chat_mock, planner_agent, workflow_state, workflow_msg_event
    ):
        workflow_state["last_human_input"] = workflow_msg_event
        simple_string = "This is a simple string message"
        chat_mock.ainvoke.return_value = AIMessage(content=simple_string)

        result = await planner_agent.run(workflow_state)

        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["content"] == simple_string

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_duo_workflow_service_container")
    async def test_run_with_list_of_strings_content(
        self, chat_mock, planner_agent, workflow_state, workflow_msg_event
    ):
        workflow_state["last_human_input"] = workflow_msg_event
        string_list: list[str | dict[Any, Any]] = ["Line 1", "Line 2", "Line 3"]
        chat_mock.ainvoke.return_value = AIMessage(content=string_list)

        result = await planner_agent.run(workflow_state)

        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["content"] == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_duo_workflow_service_container")
    async def test_run_with_dict_content(
        self, chat_mock, planner_agent, workflow_state, workflow_msg_event
    ):
        workflow_state["last_human_input"] = workflow_msg_event
        dict_list: list[str | dict[Any, Any]] = [
            {"text": "Message from dict"},
            {"other": "data"},
        ]
        chat_mock.ainvoke.return_value = AIMessage(content=dict_list)

        result = await planner_agent.run(workflow_state)

        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["content"] == "Message from dict"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_duo_workflow_service_container")
    async def test_run_with_invalid_content(
        self, chat_mock, planner_agent, workflow_state, workflow_msg_event
    ):
        workflow_state["last_human_input"] = workflow_msg_event
        invalid_content: list[str | dict[Any, Any]] = [
            {"other": "data"},
            {"more": "data"},
        ]
        chat_mock.ainvoke.return_value = AIMessage(content=invalid_content)

        result = await planner_agent.run(workflow_state)

        # No ui_chat_log entries should be created for invalid content
        assert len(result["ui_chat_log"]) == 0

    @pytest.mark.asyncio
    async def test_run_without_human_input(
        self, chat_mock, planner_agent, workflow_state
    ):
        simple_string = "This is a simple string message"
        chat_mock.ainvoke.return_value = AIMessage(content=simple_string)

        result = await planner_agent.run(workflow_state)

        assert not "ui_chat_log" in result

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_duo_workflow_service_container")
    async def test_run_with_resume_event(
        self, chat_mock, planner_agent, workflow_state, workflow_resume_event
    ):
        workflow_state["last_human_input"] = workflow_resume_event
        simple_string = "This is a simple string message"
        chat_mock.ainvoke.return_value = AIMessage(content=simple_string)

        result = await planner_agent.run(workflow_state)

        assert not "ui_chat_log" in result

    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.agent.get_event")
    async def test_run_with_cancelled_workflow(
        self, mock_get_event, chat_mock, planner_agent, workflow_state
    ):
        mock_get_event.return_value = {
            "id": "event-id",
            "event_type": WorkflowEventType.STOP,
            "message": "Workflow cancelled",
        }

        result = await planner_agent.run(workflow_state)

        # Verify the model wasn't called since workflow was cancelled
        chat_mock.ainvoke.assert_not_called()
        assert result["status"] == WorkflowStatusEnum.CANCELLED

    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.agent.get_event")
    async def test_run_with_non_cancelled_workflow(
        self, mock_get_event, chat_mock, planner_agent, workflow_state
    ):
        mock_get_event.return_value = {
            "id": "event-id",
            "event_type": "message",
            "message": "Continue working",
        }

        chat_mock.ainvoke.return_value = AIMessage(content="Working on it")

        result = await planner_agent.run(workflow_state)

        # Verify the model was called since workflow wasn't cancelled
        chat_mock.ainvoke.assert_called_once()
        assert (
            "status" not in result or result["status"] != WorkflowStatusEnum.CANCELLED
        )

    @pytest.mark.asyncio
    async def test_run_with_persistent_api_error(
        self, chat_mock, planner_agent, workflow_state
    ):
        mock_error_handler = AsyncMock(spec=ModelErrorHandler)
        mock_error_handler.handle_error.return_value = AsyncMock()
        planner_agent._error_handler = mock_error_handler

        error_response = APIStatusError(
            message="Persistent server error",
            response=MagicMock(status_code=500),
            body={"error": {"message": "Internal server error"}},
        )
        chat_mock.ainvoke = AsyncMock(side_effect=error_response)

        result = await planner_agent.run(workflow_state)

        assert result["status"] == WorkflowStatusEnum.ERROR
        assert "conversation_history" in result
        assert result["conversation_history"]["test agent"][0].content.startswith(
            "There was an error processing your request:"
        )
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["status"].value == "failure"

    @pytest.mark.asyncio
    async def test_agent_processing_error_properties(
        self, chat_mock, planner_agent, workflow_state
    ):
        mock_error_handler = AsyncMock(spec=ModelErrorHandler)
        exhausted_error = ModelError(
            error_type=ModelErrorType.API_ERROR,
            status_code=500,
            message="Test error message",
        )
        mock_error_handler.handle_error.side_effect = exhausted_error
        planner_agent._error_handler = mock_error_handler

        # Configure chat_mock.ainvoke to fail with APIStatusError
        original_error = APIStatusError(
            message="Test error message",
            response=MagicMock(status_code=500),
            body={"error": {"message": "Test error"}},
        )
        chat_mock.ainvoke = AsyncMock(side_effect=original_error)

        result = await planner_agent.run(workflow_state)

        # Verify that the error information is properly preserved
        assert result["status"] == WorkflowStatusEnum.ERROR
        assert (
            "Test error message"
            in result["conversation_history"]["test agent"][0].content
        )
        assert (
            result["ui_chat_log"][0]["content"]
            == "There was an error processing your request. Please try again or contact support if the issue persists."
        )

    # pylint: disable=too-many-positional-arguments
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.agent.get_event")
    async def test_run_with_check_events_disabled(
        self,
        mock_get_event,
        chat_mock,
        gl_http_client,
        mock_toolset,
        workflow_state,
        internal_event_client,
    ):
        # Create agent with check_events=False
        agent = Agent(
            goal="Make the world a better place",
            model=chat_mock,
            name="test agent",
            system_prompt="You are AGI entity capable of anything",
            toolset=mock_toolset,
            workflow_id="test-workflow-123",
            http_client=gl_http_client,
            workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            check_events=False,
            internal_event_client=internal_event_client,
        )

        chat_mock.ainvoke.return_value = AIMessage(
            content="Working without checking events"
        )

        result = await agent.run(workflow_state)

        # Verify get_event was not called
        mock_get_event.assert_not_called()

        # Verify the agent still processed the request
        assert "conversation_history" in result
        assert (
            result["conversation_history"]["test agent"][-1].content
            == "Working without checking events"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("mock_duo_workflow_service_container")
    @patch("duo_workflow_service.agents.agent.get_event")
    async def test_run_with_check_events_enabled(
        self,
        mock_get_event,
        chat_mock,
        gl_http_client,
        mock_toolset,
        workflow_state,
        internal_event_client,
    ):
        # Create agent with check_events=True (default)
        agent = Agent(
            goal="Make the world a better place",
            model=chat_mock,
            name="test agent",
            system_prompt="You are AGI entity capable of anything",
            toolset=mock_toolset,
            workflow_id="test-workflow-123",
            http_client=gl_http_client,
            workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            check_events=True,
            internal_event_client=internal_event_client,
        )

        # Mock event response
        mock_get_event.return_value = {
            "id": "event-id",
            "event_type": "message",
            "message": "Continue working",
        }

        chat_mock.ainvoke.return_value = AIMessage(
            content="Working with events checked"
        )

        result = await agent.run(workflow_state)

        # Verify get_event was called
        mock_get_event.assert_called_once_with(
            gl_http_client, "test-workflow-123", False
        )

        # Verify the agent processed the request
        assert "conversation_history" in result
        assert (
            result["conversation_history"]["test agent"][-1].content
            == "Working with events checked"
        )
