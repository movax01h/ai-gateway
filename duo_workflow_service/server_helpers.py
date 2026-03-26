import structlog

from contract import contract_pb2
from lib.internal_events.context import (
    current_event_context,
)

log = structlog.stdlib.get_logger("server")


def clean_start_request(start_workflow_request: contract_pb2.ClientEvent):
    request = contract_pb2.ClientEvent()
    request.CopyFrom(start_workflow_request)
    # Remove fields from being logged to prevent logging sensitive user content
    request.startRequest.ClearField("goal")
    request.startRequest.ClearField("flowConfig")
    request.startRequest.ClearField("workflowMetadata")
    request.startRequest.ClearField("additional_context")
    return request


def build_logging_context(workflow_id: str, workflow_definition: str) -> dict:
    """Build enhanced logging context with event context information."""
    event_context = current_event_context.get()

    extra_context = {
        "workflow_id": workflow_id,
        "workflow_definition": workflow_definition,
    }

    if event_context is not None:
        extra_context.update(
            {
                "instance_id": (
                    str(event_context.instance_id)
                    if event_context.instance_id is not None
                    else "None"
                ),
                "host_name": (
                    str(event_context.host_name)
                    if event_context.host_name is not None
                    else "None"
                ),
                "realm": (
                    str(event_context.realm)
                    if event_context.realm is not None
                    else "None"
                ),
                "is_gitlab_team_member": (
                    str(event_context.is_gitlab_team_member)
                    if event_context.is_gitlab_team_member is not None
                    else "None"
                ),
                "global_user_id": (
                    str(event_context.global_user_id)
                    if event_context.global_user_id is not None
                    else "None"
                ),
                "correlation_id": (
                    str(event_context.correlation_id)
                    if event_context.correlation_id is not None
                    else "None"
                ),
            }
        )
    else:
        log.debug("Event context not available for enhanced logging")

    return extra_context
