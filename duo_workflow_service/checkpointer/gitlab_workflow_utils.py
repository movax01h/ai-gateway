import base64
import json
import zlib
from enum import StrEnum
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from langgraph.checkpoint.base import Checkpoint

from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import checkpoint_decoder
from duo_workflow_service.json_encoder.encoder import CustomEncoder
from lib.internal_events.event_enum import EventPropertyEnum

STATUS_TO_EVENT_PROPERTY = {
    "finished": EventPropertyEnum.WORKFLOW_COMPLETED,
    "stopped": EventPropertyEnum.CANCELLED_BY_USER,
    "input_required": EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_INPUT,
    "plan_approval_required": EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_APPROVAL,
}


class WorkflowStatusEventEnum(StrEnum):
    START = "start"
    FINISH = "finish"
    DROP = "drop"
    RESUME = "resume"
    PAUSE = "pause"
    STOP = "stop"
    RETRY = "retry"
    REQUIRE_INPUT = "require_input"
    REQUIRE_PLAN_APPROVAL = "require_plan_approval"
    REQUIRE_TOOL_CALL_APPROVAL = "require_tool_call_approval"


SUCCESSFUL_WORKFLOW_EXECUTION_STATUSES = [
    WorkflowStatusEventEnum.FINISH,
    WorkflowStatusEventEnum.STOP,
    WorkflowStatusEventEnum.REQUIRE_INPUT,
    WorkflowStatusEventEnum.REQUIRE_PLAN_APPROVAL,
    WorkflowStatusEventEnum.REQUIRE_TOOL_CALL_APPROVAL,
]


def compress_checkpoint(data: Checkpoint) -> str:
    """Compress checkpoint using zlib compression and base64 encode.

    Args:
        data: The checkpoint dictionary to compress

    Returns:
        Base64-encoded compressed checkpoint string
    """
    json_str = json.dumps(dict(data), cls=CustomEncoder)
    compressed = zlib.compress(json_str.encode("utf-8"))
    return base64.b64encode(compressed).decode("utf-8")


def uncompress_checkpoint(compressed_data: str) -> dict:
    """Uncompress compressed checkpoint data and decode using checkpoint_decoder.

    Args:
        compressed_data: Base64-encoded zlib compressed string

    Returns:
        Uncompressed checkpoint dictionary with decoded objects
    """
    decoded = base64.b64decode(compressed_data.encode("utf-8"))
    uncompressed = zlib.decompress(decoded)
    return json.loads(uncompressed.decode("utf-8"), object_hook=checkpoint_decoder)


NOOP_WORKFLOW_STATUSES = [WorkflowStatusEnum.APPROVAL_ERROR]

BILLABLE_STATUSES = frozenset(
    [
        WorkflowStatusEnum.FINISHED,
        WorkflowStatusEnum.STOPPED,
        WorkflowStatusEnum.INPUT_REQUIRED,
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
    ]
)

# Maps current checkpoint status to the rails workflow state machine's event (if applicable)
CHECKPOINT_STATUS_TO_STATUS_EVENT = {
    "FINISHED": WorkflowStatusEventEnum.FINISH,
    "FAILED": WorkflowStatusEventEnum.DROP,
    "STOPPED": WorkflowStatusEventEnum.STOP,
    "PAUSED": WorkflowStatusEventEnum.PAUSE,
    "INPUT_REQUIRED": WorkflowStatusEventEnum.REQUIRE_INPUT,
    "PLAN_APPROVAL_REQUIRED": WorkflowStatusEventEnum.REQUIRE_PLAN_APPROVAL,
    "TOOL_CALL_APPROVAL_REQUIRED": WorkflowStatusEventEnum.REQUIRE_TOOL_CALL_APPROVAL,
}

# Maps WorkflowStatus(status key in LangGraph's WorkflowState) to checkpoint status.
# Checkpoint status represents status human-readable workflow status (displayed in the UI)
WORKFLOW_STATUS_TO_CHECKPOINT_STATUS = {
    **{
        WorkflowStatusEnum.EXECUTION: "RUNNING",
        WorkflowStatusEnum.ERROR: "FAILED",
        WorkflowStatusEnum.INPUT_REQUIRED: "INPUT_REQUIRED",
        WorkflowStatusEnum.PLANNING: "RUNNING",
        WorkflowStatusEnum.PAUSED: "PAUSED",
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED: "PLAN_APPROVAL_REQUIRED",
        WorkflowStatusEnum.NOT_STARTED: "CREATED",
        WorkflowStatusEnum.COMPLETED: "FINISHED",
        WorkflowStatusEnum.CANCELLED: "STOPPED",
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED: "TOOL_CALL_APPROVAL_REQUIRED",
    },
    **dict.fromkeys(NOOP_WORKFLOW_STATUSES, "RUNNING"),
}


def add_compression_param(endpoint: str) -> str:
    """Add compression parameter to URL query string.

    Args:
        endpoint: URL string to modify

    Returns:
        URL with accept_compressed=true parameter added
    """
    parsed = urlparse(endpoint)
    params = parse_qs(parsed.query)
    params["accept_compressed"] = ["true"]

    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))
