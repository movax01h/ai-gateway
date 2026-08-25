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
    WORKFLOW_FINISH_SUCCESS = "request_duo_workflow_success"
    WORKFLOW_FINISH_FAILURE = "request_duo_workflow_failure"
    WORKFLOW_TOOL_FAILURE = "duo_workflow_tool_failure"
    WORKFLOW_TOOL_SUCCESS = "duo_workflow_tool_success"
    WORKFLOW_TOOL_APPROVAL_REQUESTED = "request_duo_workflow_tool_approval"
    WORKFLOW_TOOL_APPROVAL_RESOLVED = "resolve_duo_workflow_tool_approval"
    WORKFLOW_TOOL_BLOCKED = "block_denied_duo_workflow_tool"
    WORKFLOW_ROUTE_DECISION = "duo_workflow_flow_route_decision"
    WORKFLOW_RESPONSE_SCHEMA_OUTPUT = "duo_workflow_response_schema_output"
    ORBIT_DAP_TOOL_CALLED = "orbit_dap_tool_called"
    ORBIT_DAP_TOOL_FAILED = "orbit_dap_tool_failed"
    ORBIT_DAP_SESSION_SUMMARY = "orbit_dap_session_summary"
    COMPACTION_EXECUTED = "duo_workflow_compaction_executed"
    LEGACY_TRIM_EXECUTED = "duo_workflow_legacy_trim_executed"
    WORKFLOW_MAX_CYCLES_REACHED = "duo_workflow_max_cycles_reached"
    WORKFLOW_MERGE_REQUEST_CREATED = "duo_workflow_merge_request_created"
    WORKFLOW_SUBAGENT_DELEGATED = "duo_workflow_subagent_delegated"
    WORKFLOW_SUBAGENT_RETURNED = "duo_workflow_subagent_returned"
    WORKFLOW_SUBAGENT_DELEGATION_REJECTED = "duo_workflow_subagent_delegation_rejected"


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

    WORKFLOW_TOOL_APPROVAL_APPROVAL = "approval"
    WORKFLOW_TOOL_APPROVAL_REJECTION = "rejection"
    WORKFLOW_TOOL_APPROVAL_MODIFICATION = "modification"


# Deprecated: Use FlowType from duo_workflow_service.entities.flow instead
class CategoryEnum(StrEnum):
    WORKFLOW_SOFTWARE_DEVELOPMENT = "software_development"
    WORKFLOW_CONVERT_TO_GITLAB_CI = "convert_to_gitlab_ci"
    WORKFLOW_CHAT = "chat"
    WORKFLOW_ISSUE_TO_MERGE_REQUEST = "issue_to_merge_request"
    CODE_REVIEW = "code_review"
    FIX_PIPELINE = "fix_pipeline"
    RESOLVE_SAST_VULNERABILITY = "resolve_sast_vulnerability"
    SAST_FP_DETECTION = "sast_fp_detection"
    SECRETS_FP_DETECTION = "secrets_fp_detection"
    AI_CATALOG_AGENT = "ai_catalog_agent"
    UNKNOWN = "unknown"
    DEVELOPER = "developer"
    CONVERT_TO_GITLAB_CI = "convert_to_gl_ci"
