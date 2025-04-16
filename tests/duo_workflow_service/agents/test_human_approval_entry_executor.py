import os
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.agents import HumanApprovalEntryExecutor
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    Plan,
    ToolStatus,
    WorkflowState,
    WorkflowStatusEnum,
)


class TestHumanApprovalEntryExecutor:
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
    async def test_run(self, workflow_state):
        executor = HumanApprovalEntryExecutor("agent", "123")

        result = await executor.run(workflow_state)

        assert result["status"] == WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED
        assert len(result["ui_chat_log"]) == 1
        assert result["ui_chat_log"][0]["message_type"] == MessageTypeEnum.REQUEST
        assert result["ui_chat_log"][0]["status"] == ToolStatus.SUCCESS

    @patch.dict(os.environ, {"USE_MEMSAVER": "true"})
    @pytest.mark.asyncio
    async def test_run_with_diabled_interrupts(self, workflow_state):
        executor = HumanApprovalEntryExecutor("agent", "123")

        result = await executor.run(workflow_state)

        assert result["status"] == WorkflowStatusEnum.PLANNING
