from datetime import datetime, timezone
from typing import Any, Dict, Union

import structlog
from langgraph.types import StateSnapshot

from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    TaskStatus,
    ToolStatus,
    UiChatLog,
    WorkflowState,
)

FINISHED_STATUSES = [TaskStatus.COMPLETED, TaskStatus.CANCELLED]


class PlanTerminatorAgent:
    _workflow_id: str

    def __init__(self, workflow_id: str):
        self._workflow_id = workflow_id
        self.log = structlog.stdlib.get_logger("workflow").bind(workflow_id=workflow_id)

    async def run(self, state: Union[StateSnapshot, WorkflowState]) -> Dict[str, Any]:
        state_dict = state.values if isinstance(state, StateSnapshot) else state

        if state_dict.get("plan") is None or "steps" not in state_dict["plan"]:
            return {"plan": {"steps": []}}

        needs_updates = any(
            step["status"] not in FINISHED_STATUSES
            for step in state_dict["plan"]["steps"]
        )

        if not needs_updates:
            return {"plan": state_dict.get("plan", {})}

        updated_steps = []
        for step in state_dict["plan"]["steps"]:
            step_copy = step.copy()
            if step_copy["status"] not in FINISHED_STATUSES:
                step_copy["status"] = TaskStatus.CANCELLED
            updated_steps.append(step_copy)

        message = "Your request was valid but Workflow failed to complete it. Please try again."

        self.log.info(f"PlanTerminator: {message}")

        return {
            "plan": {"steps": updated_steps},
            "ui_chat_log": [
                UiChatLog(
                    message_type=MessageTypeEnum.WORKFLOW_END,
                    content=message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.FAILURE,
                    correlation_id=None,
                    tool_info=None,
                    context_elements=None,
                )
            ],
        }
