from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

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
    ToolStatus,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.internal_events import InternalEventAdditionalProperties
from duo_workflow_service.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventPropertyEnum,
)
from duo_workflow_service.tools import Toolset


@pytest.fixture
def workflow_msg_event():
    return {
        "id": "event-id",
        "event_type": "message",
        "message": "this is a test message",
    }


@pytest.fixture
def workflow_resume_event():
    return {
        "id": "event-id",
        "event_type": "resume",
        "message": "",
    }


@pytest.fixture
def workflow_state():
    return WorkflowState(
        plan=Plan(steps=[]),
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        handover=[],
        last_human_input=None,
        ui_chat_log=[],
    )


class TestAgent:
    @pytest.fixture
    def chat_mock(self):
        mock = MagicMock(BaseChatModel)
        mock.bind_tools.return_value = mock
        return mock

    @pytest.fixture
    def http_client_mock(self):
        return MagicMock(spec=GitlabHttpClient)

    @pytest.fixture
    def mock_toolset(self):
        mock = MagicMock(spec=Toolset)
        mock.bindable = []
        return mock

    @pytest.fixture
    def planner_agent(self, chat_mock, http_client_mock, mock_toolset):
        return Agent(
            goal="Make the world a better place",
            model=chat_mock,
            name="test agent",
            system_prompt="You are AGI entity capable of anything",
            toolset=mock_toolset,
            workflow_id="test-workflow-123",
            http_client=http_client_mock,
            workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        )

    def test_init(self, chat_mock, planner_agent, http_client_mock, mock_toolset):
        assert planner_agent._goal == "Make the world a better place"
        assert planner_agent._model == chat_mock
        assert planner_agent.name == "test agent"
        assert planner_agent._system_prompt == "You are AGI entity capable of anything"
        assert planner_agent._workflow_id == "test-workflow-123"
        assert planner_agent._http_client == http_client_mock
        assert planner_agent._toolset == mock_toolset
        chat_mock.bind_tools.assert_called_once_with(mock_toolset.bindable)

    @pytest.mark.asyncio
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
    @patch("duo_workflow_service.agents.agent.DuoWorkflowInternalEvent")
    async def test_run(
        self,
        mock_internal_event_tracker,
        chat_mock,
        planner_agent,
        workflow_state,
    ):
        workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
        mock_internal_event_tracker.instance = MagicMock(return_value=None)
        mock_internal_event_tracker.track_event = MagicMock(return_value=None)

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

        assert mock_internal_event_tracker.track_event.call_count == 1
        mock_internal_event_tracker.track_event.assert_has_calls(
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
    async def test_run_with_api_error_retry(
        self, chat_mock, planner_agent, workflow_state
    ):
        # Configure chat_mock.ainvoke to fail with APIStatusError first, then succeed
        error_response = APIStatusError(
            message="Temporary server error",
            response=MagicMock(status_code=500),
            body={"error": {"message": "Service temporarily unavailable"}},
        )

        success_response = AIMessage(content="Success after retry!")
        chat_mock.ainvoke = AsyncMock(side_effect=[error_response, success_response])

        result = await planner_agent.run(workflow_state)

        # Verify that ainvoke was called twice - first failing, then succeeding
        assert chat_mock.ainvoke.call_count == 2
        # Verify the successful response was returned
        assert (
            result["conversation_history"]["test agent"][-1].content
            == "Success after retry!"
        )

    @pytest.mark.asyncio
    async def test_run_with_persistent_api_error(
        self, chat_mock, planner_agent, workflow_state
    ):
        # Configure the error handler class to return mock instance
        mock_error_handler = AsyncMock()
        mock_error_handler.handle_error.return_value = AsyncMock()
        planner_agent._error_handler = mock_error_handler

        # Configure chat_mock.ainvoke to consistently fail with APIStatusError
        error_response = APIStatusError(
            message="Persistent server error",
            response=MagicMock(status_code=500),
            body={"error": {"message": "Internal server error"}},
        )
        # Mock to return error response for all retries (3 attempts)
        chat_mock.ainvoke = AsyncMock(side_effect=[error_response] * 3)

        with pytest.raises(StopAsyncIteration):
            await planner_agent.run(workflow_state)

        # Verify that ainvoke was called the maximum number of retries + 1 (4 times)
        assert chat_mock.ainvoke.call_count == 4
        # Verify that error handler was called the maximum number of retries (3 times)
        assert mock_error_handler.handle_error.call_count == 3

    @pytest.mark.asyncio
    async def test_run_with_api_error_status_tracking(
        self, chat_mock, planner_agent, workflow_state
    ):
        # Create different API errors with various status codes
        error_429 = APIStatusError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit reached"}},
        )
        error_500 = APIStatusError(
            message="Server error",
            response=MagicMock(status_code=500),
            body={"error": {"message": "Internal server error"}},
        )
        error_529 = APIStatusError(
            message="Service unavailable",
            response=MagicMock(status_code=529),
            body={"error": {"message": "Too many requests"}},
        )

        # Configure the mock to fail with different errors then succeed
        success_response = AIMessage(content="Finally succeeded!")
        chat_mock.ainvoke = AsyncMock(
            side_effect=[error_429, error_500, error_529, success_response]
        )

        # Configure the error handler class to return mock instance
        mock_error_handler = AsyncMock()
        mock_error_handler.handle_error.return_value = AsyncMock()
        planner_agent._error_handler = mock_error_handler

        result = await planner_agent.run(workflow_state)

        # Verify all retries were attempted
        assert chat_mock.ainvoke.call_count == 4

        # Verify that error handler was called the maximum number of retries (3 times)
        assert mock_error_handler.handle_error.call_count == 3

        # Verify final success
        assert (
            result["conversation_history"]["test agent"][-1].content
            == "Finally succeeded!"
        )
