from copy import deepcopy
from datetime import datetime, timezone
from json import dumps
from typing import Optional, Union

import structlog
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk

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
        self.checkpoint_number = 0
        self.latest_ai_message: Optional[BaseMessageChunk] = None

    async def send_event(
        self,
        type: str,
        state: Union[dict, tuple[BaseMessage, dict]],
        stream: bool,
    ):
        # We must increment the checkpoint_number with every new outgoing
        # message. This value is used in conjunction with
        # most_recent_new_checkpoint to ensure we skip old checkpoints.
        self.checkpoint_number += 1

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
        # This is a placeholder empty message. The message will be replaced
        # with most_recent_new_checkpoint below
        action = contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(),
        )

        log = structlog.stdlib.get_logger("workflow")
        log.info("Attempting to add NewCheckpoint to outbox")

        self.outbox.put_action(action)

        log.info("Added NewCheckpoint to outbox")

    def most_recent_checkpoint_number(self):
        return self.checkpoint_number

    # The most_recent_new_checkpoint is an optimization for streaming
    # checkpoints to clients. Without this optimization we waste a lot of CPU
    # and bandwidth sending every incremental change to clients. This
    # optimization works by always finding the latest one when there are
    # multiple checkpoints in the outbox. For example if the outbox contains 10
    # newCheckpoint messages, when we go to send to the client, we can skip 9 of
    # them and just send the most recent. This is done by keeping track of the
    # checkpoint_number so we remember the last one we sent.
    def most_recent_new_checkpoint(self):
        return contract_pb2.NewCheckpoint(
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
        )

    def _append_chunk_to_ui_chat_log(self, message: BaseMessage):
        """Append a message chunk to the UI chat log.

        Processes incoming message chunks and either creates a new chat log entry
        or appends content to the existing last entry if it's an ongoing agent message.

        Args:
            message (BaseMessage): The message chunk to be processed and added to the log.
        """

        if not isinstance(message, AIMessageChunk):
            return

        if self.latest_ai_message and self.latest_ai_message.id == message.id:
            self.latest_ai_message += message
            self.ui_chat_log[-1]["content"] = self.latest_ai_message.text()
        else:
            self.latest_ai_message = message
            last_ui_message = UiChatLog(
                message_id=message.id,
                status=None,
                correlation_id=None,
                message_type=MessageTypeEnum.AGENT,
                message_sub_type=None,
                timestamp=datetime.now(timezone.utc).isoformat(),
                content=message.text(),
                tool_info=None,
                additional_context=None,
            )
            self.ui_chat_log.append(last_ui_message)
