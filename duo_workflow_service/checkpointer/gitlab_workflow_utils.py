from duo_workflow_service.internal_events.event_enum import EventPropertyEnum

STATUS_TO_EVENT_PROPERTY = {
    "finished": EventPropertyEnum.WORKFLOW_COMPLETED,
    "stopped": EventPropertyEnum.CANCELLED_BY_USER,
    "input_required": EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_INPUT,
    "plan_approval_required": EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_APPROVAL,
}
