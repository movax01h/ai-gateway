from copy import deepcopy
from datetime import datetime, timezone
from json import dumps
from typing import Union

import structlog
from langchain_core.messages import BaseMessage

from contract import contract_pb2
from duo_workflow_service.checkpointer.gitlab_workflow import (
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
)
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.json_encoder.encoder import CustomEncoder


class UserInterface:
    def __init__(
        self,
        outbox: Outbox,
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

            has_content = self._append_chunk_to_ui_chat_log(message)
            if has_content:
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
                    },
                    cls=CustomEncoder,
                ),
            ),
        )

        log = structlog.stdlib.get_logger("workflow")
        log.info("Attempting to add NewCheckpoint to outbox")

        self.outbox.put_action(action)

        log.info("Added NewCheckpoint to outbox")

    def _append_chunk_to_ui_chat_log(self, message: BaseMessage) -> bool:
        """Append a message chunk to the UI chat log.

        Processes incoming message chunks and either creates a new chat log entry
        or appends content to the existing last entry if it's an ongoing agent message.

        Args:
            message (BaseMessage): The message chunk to be processed and added to the log.

        Returns:
            bool: True if content was successfully added to the chat log, False if
                the message had no content to add.
        """
        if (
            not self.ui_chat_log
            or self.ui_chat_log[-1]["message_type"] != MessageTypeEnum.AGENT
            or self.ui_chat_log[-1]["status"]
        ):
            last_message = UiChatLog(
                status=None,
                correlation_id=None,
                message_type=MessageTypeEnum.AGENT,
                message_sub_type=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content="",
                tool_info=None,
                additional_context=None,
            )
            self.ui_chat_log.append(last_message)
        else:
            last_message = self.ui_chat_log[-1]

        last_message["content"] = last_message["content"] + message.text()

        return True
