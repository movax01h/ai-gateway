# pylint: disable=direct-environment-variable-reference

import os
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agents import HumanApprovalCheckExecutor
from duo_workflow_service.entities import WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient


@pytest.fixture(autouse=True)
def prepare_container(
    mock_duo_workflow_service_container,
):  # pylint: disable=unused-argument
    pass


class TestHumanApprovalCheckExecutor:
    @pytest.fixture(name="mock_http_client")
    def mock_http_client_fixture(self):
        return AsyncMock(spec=GitlabHttpClient)

    @pytest.fixture(name="workflow_state")
    def workflow_state_fixture(self):
        return WorkflowState(
            plan=Plan(steps=[]),
            status=WorkflowStatusEnum.PLANNING,
            conversation_history={},
            handover=[],
            last_human_input=None,
            ui_chat_log=[],
        )

    @patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "true"})
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.human_approval_check_executor.interrupt")
    async def test_run_with_resume(self, mock_interrupt, workflow_state):
        event = {"event_type": WorkflowEventType.RESUME}
        mock_interrupt.return_value = event
        executor = HumanApprovalCheckExecutor("agent", "1234", "approved-agent-status")

        result = await executor.run(workflow_state)

        assert result == {
            "last_human_input": event,
            "ui_chat_log": [],
            "status": "approved-agent-status",
        }

    @patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "true"})
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.human_approval_check_executor.interrupt")
    @patch("duo_workflow_service.agents.human_approval_check_executor.uuid4")
    async def test_run_with_message(self, mock_uuid4, mock_interrupt, workflow_state):
        mock_uuid4.return_value = UUID("87654321-4321-8765-4321-876543218765")

        event = {"event_type": WorkflowEventType.MESSAGE, "message": "response"}
        mock_interrupt.return_value = event
        executor = HumanApprovalCheckExecutor("agent", "1234", "approved-agent-status")
        workflow_state["conversation_history"] = {
            "agent": [
                AIMessage(
                    content="Previous message",
                    tool_calls=[{"id": "tool_123", "name": "test", "args": {}}],
                )
            ]
        }

        result = await executor.run(workflow_state)

        assert result["status"] == "approved-agent-status"
        assert result["last_human_input"] == event
        assert len(result["ui_chat_log"]) == 1
        assert (
            result["ui_chat_log"][0]["message_id"]
            == "user-87654321-4321-8765-4321-876543218765"
        )
        assert result["conversation_history"]["agent"] == [
            ToolMessage(
                content="Tool cancelled temporarily as user has a question",
                tool_call_id="tool_123",
            ),
            HumanMessage(
                content="response", additional_kwargs={}, response_metadata={}
            ),
        ]

    @patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "true"})
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.human_approval_check_executor.interrupt")
    @patch("duo_workflow_service.agents.human_approval_check_executor.uuid4")
    async def test_run_with_empty_message(
        self, mock_uuid4, mock_interrupt, workflow_state
    ):
        mock_uuid4.return_value = UUID("12345678-1234-5678-1234-567812345678")

        event = {"event_type": WorkflowEventType.MESSAGE, "message": ""}
        mock_interrupt.return_value = event
        executor = HumanApprovalCheckExecutor("agent", "1234", "approved-agent-status")
        workflow_state["conversation_history"] = {
            "agent": [
                AIMessage(
                    content="Previous message",
                    tool_calls=[{"id": "tool_123", "name": "test", "args": {}}],
                )
            ]
        }

        result = await executor.run(workflow_state)

        assert result["status"] == "approved-agent-status"
        assert result["last_human_input"] == event
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
        assert (
            result["ui_chat_log"][0]["content"]
            == "No message received, continuing workflow"
        )
        assert (
            result["ui_chat_log"][0]["message_id"]
            == "approval-12345678-1234-5678-1234-567812345678"
        )
        assert (
            "conversation_history" not in result
        )  # Ensure no conversation history updates for empty message
