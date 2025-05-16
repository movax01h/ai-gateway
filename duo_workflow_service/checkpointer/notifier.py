import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from typing import Union

from langchain.load.dump import dumps
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.string import StrOutputParser

from contract import contract_pb2
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
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
        self.status = WorkflowStatusEnum.NOT_STARTED
        self.steps: list[dict] = []

    async def send_event(
        self,
        type: str,
        state: Union[dict, tuple[BaseMessage, dict]],
        stream: bool,
    ):
        if type == "values" and isinstance(state, dict):
            self.status = state["status"]
            self.steps = state.get("plan", {}).get("steps", [])
            self.ui_chat_log = deepcopy(state["ui_chat_log"])

            return await self._execute_action()

        if not stream:
            return

        if type == "messages":
            (message, _) = state
            self._append_chunk_to_ui_chat_log(message)

            return await self._execute_action()

    async def _execute_action(self):
        action = contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(
                goal=self.goal,
                status=WORKFLOW_STATUS_TO_CHECKPOINT_STATUS[self.status],
                checkpoint=dumps(
                    {
                        "channel_values": {
                            "ui_chat_log": self.ui_chat_log,
                            "plan": {"steps": self.steps},
                        }
                    }
                ),
            ),
        )

        return await self.outbox.put(action)

    def _append_chunk_to_ui_chat_log(self, message: BaseMessage):
        content = StrOutputParser().invoke(message) or ""
        if not content:
            return

        if (
            not self.ui_chat_log
            or self.ui_chat_log[-1]["message_type"] != MessageTypeEnum.AGENT
            or self.ui_chat_log[-1]["status"]
        ):
            last_message = UiChatLog(
                status=None,
                correlation_id=None,
                message_type=MessageTypeEnum.AGENT,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content="",
                tool_info=None,
            )
            self.ui_chat_log.append(last_message)
        else:
            last_message = self.ui_chat_log[-1]

        last_message["content"] = last_message["content"] + content
