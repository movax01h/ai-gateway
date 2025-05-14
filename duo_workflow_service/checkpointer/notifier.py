import asyncio

from langchain.load.dump import dumps

from contract import contract_pb2
from duo_workflow_service.entities.state import (
    DuoWorkflowStateType,
    UiChatLog,
    WorkflowStatusEnum,
)

WORKFLOW_STATUS_TO_CHECKPOINT_STATUS = {
    WorkflowStatusEnum.EXECUTION: "RUNNING",
    WorkflowStatusEnum.ERROR: "FAILED",
    WorkflowStatusEnum.INPUT_REQUIRED: "INPUT_REQUIRED",
    WorkflowStatusEnum.PLANNING: "RUNNING",
    WorkflowStatusEnum.PAUSED: "PAUSED",
    WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED: "PLAN_APPROVAL_REQUIRED",
    WorkflowStatusEnum.NOT_STARTED: "CREATED",
    WorkflowStatusEnum.COMPLETED: "FINISHED",
    WorkflowStatusEnum.CANCELLED: "STOPPED",
    WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED: "REQUIRE_TOOL_CALL_APPROVAL",
}


class UserInterface:
    def __init__(
        self,
        outbox: asyncio.Queue,
        goal: str,
    ):
        self.outbox = outbox
        self.goal = goal
        self.ui_chat_log: list[UiChatLog] = []

    async def send_event(self, type: str, state: DuoWorkflowStateType):
        if type != "values":
            return

        self.ui_chat_log = state["ui_chat_log"]

        return await self._execute_action(state)

    async def _execute_action(self, state: DuoWorkflowStateType):
        action = contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(
                goal=self.goal,
                status=WORKFLOW_STATUS_TO_CHECKPOINT_STATUS[state["status"]],
                checkpoint=dumps(
                    {
                        "channel_values": {
                            "ui_chat_log": self.ui_chat_log,
                            "plan": {"steps": state.get("plan", {}).get("steps", None)},
                        }
                    }
                ),
            )
        )

        return await self.outbox.put(action)
