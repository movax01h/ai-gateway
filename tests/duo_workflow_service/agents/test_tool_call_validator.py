from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agents.tool_call_validator import (
    MAX_MALFORMED_TOOL_CALL_RETRIES,
    retry_malformed_tool_calls,
    validate_tool_calls,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.tools import MalformedToolCallError
from duo_workflow_service.tools.toolset import Toolset


@pytest.fixture(name="mock_toolset")
def mock_toolset_fixture():
    mock = Mock(spec=Toolset)
    mock.validate_tool_call.return_value = None
    return mock


@pytest.fixture(name="state")
def state_fixture():
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


class TestValidateToolCalls:
    """Test validate_tool_calls function."""

    def test_valid_tool_calls_returns_empty_list(self, mock_toolset):
        """Valid tool calls should return no errors."""
        message = AIMessage(
            content="I'll create the commit",
            tool_calls=[
                {
                    "name": "create_commit",
                    "args": {
                        "actions": [
                            {
                                "action": "create",
                                "file_path": "test.py",
                                "content": "hi",
                            }
                        ]
                    },
                    "id": "call_valid",
                    "type": "tool_call",
                }
            ],
        )

        result = validate_tool_calls(mock_toolset, message)

        assert not result
        mock_toolset.validate_tool_call.assert_called_once()

    def test_malformed_tool_call_returns_error_messages(self, mock_toolset):
        """Malformed tool calls should return ToolMessage errors."""
        malformed_call = {
            "name": "create_commit",
            "args": {"actions": "not a list"},
            "id": "call_bad",
            "type": "tool_call",
        }
        message = AIMessage(
            content="I'll create the commit",
            tool_calls=[malformed_call],
        )
        mock_toolset.validate_tool_call.side_effect = MalformedToolCallError(
            "Invalid arguments",
            tool_call=malformed_call,
        )

        result = validate_tool_calls(mock_toolset, message)

        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].tool_call_id == "call_bad"
        assert "Invalid arguments" in result[0].content

    def test_mixed_valid_and_malformed_calls(self, mock_toolset):
        """Only malformed calls should produce error messages."""
        valid_call = {
            "name": "read_file",
            "args": {"path": "test.py"},
            "id": "call_good",
            "type": "tool_call",
        }
        malformed_call = {
            "name": "create_commit",
            "args": {"actions": "bad"},
            "id": "call_bad",
            "type": "tool_call",
        }
        message = AIMessage(
            content="I'll do both",
            tool_calls=[valid_call, malformed_call],
        )
        mock_toolset.validate_tool_call.side_effect = [
            None,
            MalformedToolCallError("Invalid arguments", tool_call=malformed_call),
        ]

        result = validate_tool_calls(mock_toolset, message)

        assert len(result) == 1
        assert result[0].tool_call_id == "call_bad"


class TestRetryMalformedToolCalls:
    """Test retry_malformed_tool_calls function."""

    @pytest.mark.asyncio
    async def test_successful_retry(self, mock_toolset, state):
        """When LLM produces malformed tool call args, retry should return the corrected call."""
        malformed_call = {
            "name": "create_commit",
            "args": {"actions": "not a list"},
            "id": "call_bad",
            "type": "tool_call",
        }
        corrected_call = {
            "name": "create_commit",
            "args": {
                "actions": [
                    {
                        "action": "create",
                        "file_path": "test.py",
                        "content": "print('hi')",
                    }
                ]
            },
            "id": "call_good",
            "type": "tool_call",
        }

        malformed_response = AIMessage(
            content="I'll create the commit",
            tool_calls=[malformed_call],
            id="msg-bad",
        )
        corrected_response = AIMessage(
            content="I'll create the commit",
            tool_calls=[corrected_call],
            id="msg-good",
        )

        mock_get_response = AsyncMock(return_value=corrected_response)
        validation_errors = [
            ToolMessage(content="Invalid arguments", tool_call_id="call_bad")
        ]

        # After retry, validation succeeds
        mock_toolset.validate_tool_call.return_value = None

        result = await retry_malformed_tool_calls(
            toolset=mock_toolset,
            agent_response=malformed_response,
            validation_errors=validation_errors,
            state=state,
            agent_name="Chat Agent",
            get_agent_response=mock_get_response,
        )

        assert result == corrected_response
        mock_get_response.assert_called_once()

        # Verify the malformed response + error were appended to history
        history = state["conversation_history"]["Chat Agent"]
        assert any(isinstance(msg, ToolMessage) for msg in history)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_returns_error(self, mock_toolset, state):
        """After MAX_MALFORMED_TOOL_CALL_RETRIES, should return error message."""
        malformed_call = {
            "name": "create_commit",
            "args": {"actions": "bad"},
            "id": "call_bad",
            "type": "tool_call",
        }
        malformed_response = AIMessage(
            content="I'll create the commit",
            tool_calls=[malformed_call],
            id="msg-bad",
        )

        mock_get_response = AsyncMock(return_value=malformed_response)
        validation_errors = [
            ToolMessage(content="Invalid arguments", tool_call_id="call_bad")
        ]
        mock_toolset.validate_tool_call.side_effect = MalformedToolCallError(
            "Invalid arguments",
            tool_call=malformed_call,
        )

        result = await retry_malformed_tool_calls(
            toolset=mock_toolset,
            agent_response=malformed_response,
            validation_errors=validation_errors,
            state=state,
            agent_name="Chat Agent",
            get_agent_response=mock_get_response,
        )

        assert isinstance(result, AIMessage)
        assert (
            "encountered a problem making a call to the create_commit tool"
            in result.content
        )
        assert mock_get_response.call_count == MAX_MALFORMED_TOOL_CALL_RETRIES
