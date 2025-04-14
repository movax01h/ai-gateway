from enum import StrEnum
from typing import Optional, TypedDict


class WorkflowEventType(StrEnum):
    RESPONSE = "response"
    MESSAGE = "message"
    PAUSE = "pause"
    STOP = "stop"
    RESUME = "resume"
    REQUIRE_INPUT = "require_input"


class WorkflowEvent(TypedDict):
    id: str
    event_type: WorkflowEventType
    message: str
    correlation_id: Optional[str]
