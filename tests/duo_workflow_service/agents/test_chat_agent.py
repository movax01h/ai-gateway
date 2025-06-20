from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)


@pytest.fixture
def sample_input_state():
    return {
        "conversation_history": {"test_agent": []},
        "project": None,
    }


class TestChatAgent:
    @pytest.mark.asyncio
    async def test_run_success_with_tool_calls(self, sample_input_state):
        mock_ai_message = AIMessage(
            content="Test response",
            tool_calls=[
                {
                    "name": "test_tool",
                    "args": {"param": "value"},
                    "id": "test_tool_call_1",
                }
            ],
        )
        agent = Mock(spec=ChatAgent)
        agent.name = "test_agent"
        agent.ainvoke = AsyncMock(return_value=mock_ai_message)
        agent._get_approvals = Mock(return_value=(False, []))

        with patch(
            "duo_workflow_service.agents.chat_agent.Prompt.ainvoke",
            new_callable=AsyncMock,
        ) as mock_super_ainvoke:
            mock_super_ainvoke.return_value = mock_ai_message

            result = await ChatAgent.run(agent, sample_input_state)

        assert result["status"] == WorkflowStatusEnum.EXECUTION
        assert result["conversation_history"]["test_agent"] == [mock_ai_message]
        assert "ui_chat_log" not in result

    @pytest.mark.asyncio
    async def test_run_success_without_tool_calls(self, sample_input_state):
        mock_ai_message = AIMessage(content="Test response without tools")

        agent = Mock(spec=ChatAgent)
        agent.name = "test_agent"

        agent.ainvoke = AsyncMock(return_value=mock_ai_message)

        with patch(
            "duo_workflow_service.agents.chat_agent.Prompt.ainvoke",
            new_callable=AsyncMock,
        ) as mock_super_ainvoke:
            mock_super_ainvoke.return_value = mock_ai_message

            result = await ChatAgent.run(agent, sample_input_state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert result["conversation_history"]["test_agent"] == [mock_ai_message]
        assert "ui_chat_log" in result
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert result["ui_chat_log"][0]["content"] == "Test response without tools"
        assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_run_exception_handling(self, sample_input_state):
        test_error = Exception("Test error message")

        agent = Mock(spec=ChatAgent)
        agent.name = "test_agent"

        agent.ainvoke = AsyncMock(side_effect=test_error)

        with patch(
            "duo_workflow_service.agents.chat_agent.Prompt.ainvoke",
            new_callable=AsyncMock,
        ) as mock_super_ainvoke:
            mock_super_ainvoke.side_effect = test_error

            result = await ChatAgent.run(agent, sample_input_state)

        assert result["status"] == WorkflowStatusEnum.INPUT_REQUIRED
        assert len(result["conversation_history"]["test_agent"]) == 1

        error_message = result["conversation_history"]["test_agent"][0]
        assert isinstance(error_message, HumanMessage)
        assert (
            "There was an error processing your request: Test error message"
            in error_message.content
        )

        assert "ui_chat_log" in result
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert (
            "There was an error processing your request. Please try again or contact support if the issue persists."
            in result["ui_chat_log"][0]["content"]
        )
        assert result["ui_chat_log"][0]["status"] == ToolStatus.FAILURE
