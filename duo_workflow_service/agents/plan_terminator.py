from typing import Any

import structlog

from duo_workflow_service.entities.state import TaskStatus, WorkflowState

FINISHED_STATUSES = [TaskStatus.COMPLETED, TaskStatus.CANCELLED]


class PlanTerminatorAgent:
    _workflow_id: str

    def __init__(self, workflow_id: str):
        self._workflow_id = workflow_id
        self.log = structlog.stdlib.get_logger("workflow").bind(workflow_id=workflow_id)

    async def run(self, state: WorkflowState) -> dict[str, Any]:
        if state.get("plan") is None or "steps" not in state["plan"]:
            return {"plan": {"steps": []}}

        needs_updates = any(
            step["status"] not in FINISHED_STATUSES for step in state["plan"]["steps"]
        )

        if not needs_updates:
            return {"plan": state.get("plan", {})}

        updated_steps = []
        for step in state["plan"]["steps"]:
            step_copy = step.copy()
            if step_copy["status"] not in FINISHED_STATUSES:
                step_copy["status"] = TaskStatus.CANCELLED
            updated_steps.append(step_copy)

        message = "Your request was valid but Workflow failed to complete it. Please try again."

        self.log.info(f"PlanTerminator: {message}")

        return {
            "plan": {"steps": updated_steps},
        }
