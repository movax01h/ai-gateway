from enum import StrEnum


class EventEnum(StrEnum):
    WORKFLOW_START = "request_duo_workflow"
    WORKFLOW_RESUME = "resume_duo_workflow"
    WORKFLOW_PAUSE = "pause_duo_workflow"
    WORKFLOW_RETRY = "retry_request_duo_workflow"
    WORKFLOW_MESSAGE = "message_duo_workflow"
    TOKEN_PER_USER_PROMPT = "tokens_per_user_request_prompt"
    WORKFLOW_FINISH_SUCCESS = "request_duo_workflow_success"
    WORKFLOW_FINISH_FAILURE = "request_duo_workflow_failure"
    WORKFLOW_TOOL_FAILURE = "duo_workflow_tool_failure"
    WORKFLOW_TOOL_SUCCESS = "duo_workflow_tool_success"


class EventLabelEnum(StrEnum):
    WORKFLOW_FINISH_LABEL = "workflow_finish_event"
    WORKFLOW_START_LABEL = "workflow_start_event"
    WORKFLOW_TOOL_CALL_LABEL = "workflow_tool_call"
    WORKFLOW_RESUME_LABEL = "workflow_resume_event"
    WORKFLOW_PAUSE_LABEL = "workflow_pause_event"
    WORKFLOW_MESSAGE_LABEL = "workflow_message_event"


class EventPropertyEnum(StrEnum):
    WORKFLOW_ID = "workflow_id"
    CANCELLED_BY_USER = "cancelled_by_user"
    WORKFLOW_COMPLETED = "workflow_completed"

    WORKFLOW_RESUME_BY_USER = "resume_request_by_user_duo_workflow"
    WORKFLOW_RESUME_BY_PLAN = "resume_request_by_agent_duo_workflow"

    WORKFLOW_PAUSE_BY_USER = "pause_request_by_user_duo_workflow"
    WORKFLOW_PAUSE_BY_PLAN = "pause_request_by_agent_duo_workflow"

    WORKFLOW_MESSAGE_BY_USER = "message_request_by_user_duo_workflow"
