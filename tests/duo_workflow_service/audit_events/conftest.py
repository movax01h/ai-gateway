from duo_workflow_service.audit_events.event_types import ToolInvokedEvent


def make_audit_event(tool_name="read_file", workflow_id="wf-1"):
    return ToolInvokedEvent(workflow_id=workflow_id, tool_name=tool_name)
