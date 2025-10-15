from enum import StrEnum


class EventEnum(StrEnum):
    RECEIVE_START_REQUEST = "receive_start_duo_workflow"
    WORKFLOW_START = "request_duo_workflow"
    WORKFLOW_RESUME = "resume_duo_workflow"
    WORKFLOW_PAUSE = "pause_duo_workflow"
    WORKFLOW_STOP = "stop_duo_workflow"
    WORKFLOW_RETRY = "retry_request_duo_workflow"
    WORKFLOW_MESSAGE = "message_duo_workflow"
    WORKFLOW_ABORTED = "request_duo_workflow_aborted"
    WORKFLOW_REJECT = "reject_duo_agent_platform"
    TOKEN_PER_USER_PROMPT = "tokens_per_user_request_prompt"
    WORKFLOW_FINISH_SUCCESS = "request_duo_workflow_success"
    WORKFLOW_FINISH_FAILURE = "request_duo_workflow_failure"
    WORKFLOW_TOOL_FAILURE = "duo_workflow_tool_failure"
    WORKFLOW_TOOL_SUCCESS = "duo_workflow_tool_success"


class EventLabelEnum(StrEnum):
    WORKFLOW_RECEIVE_START_REQUEST_LABEL = "workflow_receive_start_event"
    WORKFLOW_FINISH_LABEL = "workflow_finish_event"
    WORKFLOW_START_LABEL = "workflow_start_event"
    WORKFLOW_TOOL_CALL_LABEL = "workflow_tool_call"
    WORKFLOW_RESUME_LABEL = "workflow_resume_event"
    WORKFLOW_PAUSE_LABEL = "workflow_pause_event"
    WORKFLOW_MESSAGE_LABEL = "workflow_message_event"
    WORKFLOW_REJECT_LABEL = "workflow_reject_event"


class EventPropertyEnum(StrEnum):
    WORKFLOW_ID = "workflow_id"
    CANCELLED_BY_USER = "cancelled_by_user"
    WORKFLOW_COMPLETED = "workflow_completed"

    WORKFLOW_RESUME_BY_USER = "resume_request_by_user_duo_workflow"
    WORKFLOW_RESUME_BY_PLAN = "resume_request_by_agent_duo_workflow"
    WORKFLOW_RESUME_BY_PLAN_AFTER_INPUT = (
        "resume_request_by_agent_duo_workflow_after_input"
    )
    WORKFLOW_RESUME_BY_PLAN_AFTER_APPROVAL = (
        "resume_request_by_agent_duo_workflow_after_approval"
    )

    WORKFLOW_PAUSE_BY_USER = "pause_request_by_user_duo_workflow"
    WORKFLOW_PAUSE_BY_PLAN = "pause_request_by_agent_duo_workflow"
    WORKFLOW_PAUSE_BY_PLAN_AWAIT_INPUT = (
        "pause_request_by_agent_duo_workflow_await_input"
    )
    WORKFLOW_PAUSE_BY_PLAN_AWAIT_APPROVAL = (
        "pause_request_by_agent_duo_workflow_await_approval"
    )

    WORKFLOW_MESSAGE_BY_USER = "message_request_by_user_duo_workflow"


class CategoryEnum(StrEnum):
    WORKFLOW_SOFTWARE_DEVELOPMENT = "software_development"
    WORKFLOW_CONVERT_TO_GITLAB_CI = "convert_to_gitlab_ci"
    WORKFLOW_CHAT = "chat"
    WORKFLOW_ISSUE_TO_MERGE_REQUEST = "issue_to_merge_request"
    CODE_REVIEW = "code_review"
    FIX_PIPELINE = "fix_pipeline"
    RESOLVE_SAST_VULNERABILITY = "resolve_sast_vulnerability"
    SAST_FP_DETECTION = "sast_fp_detection"
    AI_CATALOG_AGENT = "ai_catalog_agent"
    UNKNOWN = "unknown"
