import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.agents import HumanApprovalCheckExecutor
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


class TestHumanApprovalCheckExecutor:
    @pytest.fixture
    def mock_http_client(self):
        return AsyncMock(spec=GitlabHttpClient)

    @pytest.fixture
    def workflow_state(self):
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
        executor = HumanApprovalCheckExecutor("agent", "1234")

        result = await executor.run(workflow_state)

        assert result == {"last_human_input": event, "ui_chat_log": []}

    @patch.dict(os.environ, {"WORKFLOW_INTERRUPT": "true"})
    @pytest.mark.asyncio
    @patch("duo_workflow_service.agents.human_approval_check_executor.interrupt")
    async def test_run_with_message(self, mock_interrupt, workflow_state):
        event = {"event_type": WorkflowEventType.MESSAGE, "message": "response"}
        mock_interrupt.return_value = event
        executor = HumanApprovalCheckExecutor("agent", "1234")
        workflow_state["conversation_history"] = {
            "agent": [
                AIMessage(
                    content="Previous message",
                    tool_calls=[{"id": "tool_123", "name": "test", "args": {}}],
                )
            ]
        }

        result = await executor.run(workflow_state)

        assert result["last_human_input"] == event
        assert result["conversation_history"]["agent"] == [
            ToolMessage(
                content="Tool cancelled temporarily as user has a question",
                tool_call_id="tool_123",
            ),
            HumanMessage(
                content="response", additional_kwargs={}, response_metadata={}
            ),
        ]

    @patch.dict(os.environ, {"USE_MEMSAVER": "true"})
    @pytest.mark.asyncio
    async def test_run_with_diabled_interrupts(self, workflow_state):
        executor = HumanApprovalCheckExecutor("agent", "1234")

        result = await executor.run(workflow_state)

        assert result["status"] == WorkflowStatusEnum.PLANNING
