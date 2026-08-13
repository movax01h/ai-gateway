import json
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Literal, Type
from uuid import uuid4

from langchain_core.tools import ToolException
from packaging.version import Version
from pydantic import BaseModel, Field

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action_and_get_action_response,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

__all__ = ["NotifyMeWhen", "NotifyMeWhenInput", "NotifyWhen"]

# Longest delay a notification may be scheduled with. An arbitrary bound to start
# with, to avoid unintended side effects; adjust as needed. This is server policy,
# not wire contract (see the ScheduleNotification comments in contract/contract.proto).
_MAX_DELAY_MINUTES = 24 * 60

_CONDITION_TYPE_TIMER = "timer"


class NotifyWhen(BaseModel):
    """The condition that triggers delivery.

    ``type`` is an open enum: new condition types arrive as new members with their
    own fields, at which point the default is dropped.
    """

    type: Literal["timer"] = Field(
        default="timer",
        description="The kind of condition. Only 'timer' is supported.",
    )
    delay_minutes: int = Field(
        ge=1,
        le=_MAX_DELAY_MINUTES,
        description=(
            "Whole minutes from now until the notification is delivered, "
            f"between 1 and {_MAX_DELAY_MINUTES}."
        ),
    )


class NotifyMeWhenInput(BaseModel):
    when: NotifyWhen = Field(description="When the message should be delivered.")
    message: str = Field(
        description=(
            "What to send yourself. The conversation continues, so a short "
            "reminder is enough."
        ),
    )


def _to_rfc3339(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NotifyMeWhen(DuoBaseTool):
    name: str = "notify_me_when"
    description: str = (
        "Schedule a message to be delivered back into this session at a future "
        "time. The message arrives as an ordinary user turn, so use it to remind "
        "yourself to resume work, check on something, or continue after a delay. "
        "This returns immediately: carry on with other work, or end your turn if "
        "there is nothing else to do. Delivery happens at or after the requested "
        "time, and cannot be cancelled once scheduled."
    )
    args_schema: Type[BaseModel] = NotifyMeWhenInput
    required_capability: ClassVar[frozenset[str]] = frozenset({"schedule_notification"})
    tool_version: ClassVar[Version] = Version("0.1.0")

    async def _execute(
        self,
        when: dict | NotifyWhen,
        message: str,
    ) -> str:
        condition = when if isinstance(when, NotifyWhen) else NotifyWhen(**when)

        # No rounding: delay_minutes 1 means 60 seconds from now. Delivery may drift
        # late, but never happens earlier than requested.
        deadline = _to_rfc3339(
            datetime.now(timezone.utc) + timedelta(minutes=condition.delay_minutes)
        )

        # One notification always has exactly one task_id. The v1 tool node does not
        # inject the tool call id onto the tool instance, so the id is generated here.
        task_id = str(uuid4())

        action = contract_pb2.Action(
            scheduleNotification=contract_pb2.ScheduleNotification(
                task_id=task_id,
                condition=json.dumps(
                    {"type": condition.type, "deadline": deadline},
                ),
                message=message,
            )
        )

        response = await _execute_action_and_get_action_response(
            self.metadata,  # type: ignore[arg-type]
            action,
        )

        response_type = response.WhichOneof("response_type")
        if response_type != "scheduleNotificationResponse":
            raise ToolException(
                "The client did not acknowledge the notification "
                f"(response type: {response_type}). Nothing is scheduled."
            )

        ack = response.scheduleNotificationResponse
        if not ack.accepted:
            raise ToolException(
                "The client refused to schedule the notification: "
                f"{ack.reason or 'no reason given'}. Nothing is scheduled; "
                "decide how to proceed."
            )

        return (
            f"Notification {task_id} scheduled. It will be delivered at or after "
            f"{ack.fires_at or deadline} (UTC)."
        )

    def format_display_message(
        self,
        args: NotifyMeWhenInput,
        _tool_response: Any = None,
    ) -> str:
        return f"Schedule a notification in {args.when.delay_minutes} minute(s)"
