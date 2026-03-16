import asyncio
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
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.json_encoder.encoder import CustomEncoder

log = structlog.stdlib.get_logger("notifier")

CHECKPOINT_THROTTLE_SECONDS = 0.05


class _ThrottleState:
    """Holds throttle state for checkpoint enqueuing."""

    def __init__(self):
        self.last_enqueued_at: float = 0
        self.trailing_task: Optional[asyncio.Task] = None


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
        self.last_sent_ui_message_id: Optional[str] = None
        self.current_resp_id: Optional[str] = None
        self._throttle = _ThrottleState()

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

            return await self._execute_action(throttle=False)

        if not stream:
            return

        if type == "messages":
            (message, _) = state

            self._append_chunk_to_ui_chat_log(message)

            return await self._execute_action(throttle=True)

    async def _execute_action(self, throttle: bool = False):
        # For streaming message chunks, throttle checkpoint enqueues so at most
        # one is sent per CHECKPOINT_THROTTLE_SECONDS window. This prevents
        # flooding workhorse with a checkpoint per LLM token.
        # A trailing edge task is always scheduled to ensure the final chunk
        # is delivered even if it falls within a throttle window.
        # For values events (status updates), bypass throttle and enqueue immediately.
        if not throttle:
            await self._enqueue_checkpoint()
            return

        now = asyncio.get_running_loop().time()
        elapsed = now - self._throttle.last_enqueued_at

        # Cancel any pending trailing edge task - we'll reschedule it.
        if self._throttle.trailing_task and not self._throttle.trailing_task.done():
            self._throttle.trailing_task.cancel()
            try:
                await self._throttle.trailing_task
            except asyncio.CancelledError:
                pass

        if elapsed >= CHECKPOINT_THROTTLE_SECONDS:
            self._throttle.last_enqueued_at = now
            await self._enqueue_checkpoint()
        else:
            # Within throttle window - skip but schedule a trailing edge send
            # so the last chunk is never lost.
            remaining = CHECKPOINT_THROTTLE_SECONDS - elapsed
            self._throttle.trailing_task = asyncio.create_task(
                self._enqueue_checkpoint_after(remaining)
            )

    async def _enqueue_checkpoint_after(self, delay: float):
        try:
            await asyncio.sleep(max(0, delay))
        except asyncio.CancelledError:
            return
        self._throttle.last_enqueued_at = asyncio.get_running_loop().time()
        await self._enqueue_checkpoint()

    async def _enqueue_checkpoint(self):
        # This is a placeholder empty message. The message will be replaced
        # with most_recent_new_checkpoint in send_events.
        action = contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(),
        )

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
        recent_ui_chat_log_changes = self._pop_recent_ui_chat_log_changes()

        return contract_pb2.NewCheckpoint(
            goal=self.goal,
            status=WORKFLOW_STATUS_TO_CHECKPOINT_STATUS[self.status],
            checkpoint=dumps(
                {
                    "channel_values": {
                        "ui_chat_log": recent_ui_chat_log_changes,
                        "plan": {"steps": self.steps},
                    }
                },
                cls=CustomEncoder,
            ),
        )

    def _pop_recent_ui_chat_log_changes(self) -> list[UiChatLog]:
        """Extract UI chat log messages that need to be sent or re-rendered.

        This method optimizes checkpoint updates by returning only the messages
        that have been added or updated since the last checkpoint was sent when
        the client supports incremental streaming. For clients that don't support
        this capability, it returns the full chat log.

        It tracks the last sent message ID and returns a slice of the chat log
        starting from that message (inclusive), allowing the client to re-render
        it if needed.

        Returns:
            list[UiChatLog]: A list of UI chat log messages that need to be sent.
                            Returns the full chat log if this is the first send,
                            the client doesn't support incremental streaming,
                            or an empty list if there are no messages.
        """
        if not self.ui_chat_log:
            return self.ui_chat_log

        if not is_client_capable("incremental_streaming"):
            return self.ui_chat_log

        if self.last_sent_ui_message_id:
            ui_chat_log_diff_idx = next(
                (
                    idx
                    for idx, msg in enumerate(self.ui_chat_log)
                    if msg.get("message_id", None) == self.last_sent_ui_message_id
                ),
                0,
            )
        else:
            ui_chat_log_diff_idx = 0

        self.last_sent_ui_message_id = self.ui_chat_log[-1].get("message_id", None)

        return self.ui_chat_log[ui_chat_log_diff_idx:]

    def _append_chunk_to_ui_chat_log(self, message: BaseMessage):
        """Append a message chunk to the UI chat log.

        Processes incoming message chunks and either creates a new chat log entry
        or appends content to the existing last entry if it's an ongoing agent message.

        Args:
            message (BaseMessage): The message chunk to be processed and added to the log.
        """
        if not isinstance(message, AIMessageChunk):
            return

        self._replace_langchain_id_with_open_ai_id(message)

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

    # OpenAI's response API returns the message start, and values with a resp_... ID
    #  instead of the LangChain ID. All streamed messages still contain a LangChain ID.
    # This causes the FE to not be able to match the streamed messages with the final message in the values.
    # This method ensures there is a consistent ID for the same message.
    def _replace_langchain_id_with_open_ai_id(self, message: BaseMessage):
        if message.id is None:
            return

        if message.id.startswith("resp_"):
            self.current_resp_id = message.id

        if message.id.startswith("lc_run") and self.current_resp_id:
            message.id = self.current_resp_id
