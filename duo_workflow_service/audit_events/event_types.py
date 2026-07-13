import uuid
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

CLOUDEVENT_SPEC_VERSION = "1.0"
CLOUDEVENT_SOURCE = "/duo_workflow_service"
CLOUDEVENT_DATA_CONTENT_TYPE = "application/json"


class AuditEventType(str, Enum):
    """Audit event types for CloudEvents `type` field.

    Member names must match their wire values (lowercased) since values are derived via auto().
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    AI_AGENT_SESSION_STARTED = auto()
    AI_AGENT_SESSION_ENDED = auto()
    AI_USER_INPUT_RECEIVED = auto()
    AI_LLM_INPUT_SENT = auto()
    AI_LLM_RESPONSE_RECEIVED = auto()
    AI_USER_OUTPUT_DISPLAYED = auto()
    AI_TOOL_INVOKED = auto()
    AI_TOOL_RESPONSE_RECEIVED = auto()
    AI_TOOL_EXECUTION_FAILED = auto()
    AI_TOOL_EXECUTION_RETRIED = auto()
    AI_LLM_REQUEST_FAILED = auto()


class AuditEvent(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    workflow_id: str
    sequence: Optional[int] = None

    def to_cloudevent(self) -> dict[str, Any]:
        data = self.model_dump(
            mode="json",
            exclude={"id", "event_type", "timestamp", "sequence"},
        )
        return {
            "specversion": CLOUDEVENT_SPEC_VERSION,
            "id": self.id,
            "type": self.event_type.value,
            "source": CLOUDEVENT_SOURCE,
            "subject": self.workflow_id,
            "time": self.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "datacontenttype": CLOUDEVENT_DATA_CONTENT_TYPE,
            "sequence": self.sequence,
            "data": data,
        }


class SessionStartedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_AGENT_SESSION_STARTED
    workflow_type: str
    goal: str


class SessionEndedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_AGENT_SESSION_ENDED
    status: str
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None


class UserInputReceivedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_USER_INPUT_RECEIVED
    input_type: str
    content: str
    content_length: int


class LlmInputSentEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_LLM_INPUT_SENT
    model_name: str
    prompt_content: str
    tools_bound: Optional[list[str]] = None


class LlmResponseReceivedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_LLM_RESPONSE_RECEIVED
    model_name: str
    response_content: str
    prompt_token_count: Optional[int] = None
    completion_token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None


class UserOutputDisplayedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_USER_OUTPUT_DISPLAYED
    output_type: str
    content: str
    content_length: int


class ToolInvokedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_TOOL_INVOKED
    tool_name: str
    tool_args: Optional[dict[str, Any]] = None


class ToolResponseReceivedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_TOOL_RESPONSE_RECEIVED
    tool_name: str
    response_content: str
    response_length: int


class ToolExecutionFailedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_TOOL_EXECUTION_FAILED
    tool_name: str
    error_type: str
    error_message: str


class ToolExecutionRetriedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_TOOL_EXECUTION_RETRIED
    tool_name: str
    attempt_number: int
    max_attempts: int
    previous_error: Optional[str] = None


class LlmRequestFailedEvent(AuditEvent):
    event_type: AuditEventType = AuditEventType.AI_LLM_REQUEST_FAILED
    model_name: str
    error_type: str
    error_message: str
    latency_ms: Optional[float] = None
