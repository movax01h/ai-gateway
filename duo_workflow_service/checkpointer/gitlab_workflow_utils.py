from enum import StrEnum

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
