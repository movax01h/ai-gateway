import structlog

from contract import contract_pb2
from lib.internal_events.context import (
    current_event_context,
)

log = structlog.stdlib.get_logger("server")


def clean_start_request(
    start_workflow_request: contract_pb2.ClientEvent,
) -> tuple[contract_pb2.ClientEvent, dict]:
    request = contract_pb2.ClientEvent()
    request.CopyFrom(start_workflow_request)
    # Capture derived fields before clearing sensitive content
    extra = {
        "hasFlowConfig": start_workflow_request.startRequest.HasField("flowConfig"),
    }
    # Remove fields from being logged to prevent logging sensitive user content
    request.startRequest.ClearField("goal")
    request.startRequest.ClearField("flowConfig")
    request.startRequest.ClearField("workflowMetadata")
    request.startRequest.ClearField("additional_context")
    return request, extra


def build_logging_context(workflow_id: str, workflow_definition: str) -> dict:
    """Build enhanced logging context with event context information."""
    event_context = current_event_context.get()

    extra_context = {
        "workflow_id": workflow_id,
        "workflow_definition": workflow_definition,
    }

    if event_context is not None:
        instance_id = (
            str(event_context.instance_id)
            if event_context.instance_id is not None
            else "None"
        )
        host_name = (
            str(event_context.host_name)
            if event_context.host_name is not None
            else "None"
        )
        realm = str(event_context.realm) if event_context.realm is not None else "None"
        is_gitlab_team_member = (
            str(event_context.is_gitlab_team_member)
            if event_context.is_gitlab_team_member is not None
            else "None"
        )
        global_user_id = (
            str(event_context.global_user_id)
            if event_context.global_user_id is not None
            else "None"
        )
        correlation_id = (
            str(event_context.correlation_id)
            if event_context.correlation_id is not None
            else "None"
        )
        extra_context.update(
            {
                # TODO: remove deprecated field aliases once dashboards are migrated
                "instance_id": instance_id,
                "host_name": host_name,
                "realm": realm,
                "global_user_id": global_user_id,
                # Canonical field names
                "gitlab_instance_id": instance_id,
                "gitlab_host_name": host_name,
                "gitlab_realm": realm,
                "is_gitlab_team_member": is_gitlab_team_member,
                "gitlab_global_user_id": global_user_id,
                "correlation_id": correlation_id,
            }
        )
    else:
        log.debug("Event context not available for enhanced logging")

    return extra_context
