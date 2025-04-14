import os
from datetime import datetime, timezone

import structlog

from duo_workflow_service.entities import WorkflowState
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)

log = structlog.get_logger("human_approval_entry_executor")


class HumanApprovalEntryExecutor:
    _agent_name: str
    _workflow_id: str

    def __init__(self, agent_name, workflow_id: str) -> None:
        self._agent_name = agent_name
        self._workflow_id = workflow_id

    async def run(self, state: WorkflowState):
        if not os.environ.get("WORKFLOW_INTERRUPT", False) or os.getenv("USE_MEMSAVER"):
            return {"status": state["status"]}

        ui_chat_logs = [
            UiChatLog(
                correlation_id=None,
                message_type=MessageTypeEnum.REQUEST,
                content="""On the left, review the proposed plan. Then ask questions or request changes.
                            To execute the plan, select Approve plan.""",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                tool_info=None,
            )
        ]

        return {
            "status": WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
            "ui_chat_log": ui_chat_logs,
        }
