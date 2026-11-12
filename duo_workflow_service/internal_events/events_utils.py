from typing import Tuple

from dependency_injector.wiring import Provide, inject

from ai_gateway.container import ContainerApplication
from duo_workflow_service.entities.state import ToolStatus
from duo_workflow_service.tracking.monitoring_context import current_monitoring_context
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum, EventLabelEnum, EventPropertyEnum

EVENT_MAPPING = {
    "require_input": (
        EventEnum.WORKFLOW_PAUSE,
        EventLabelEnum.WORKFLOW_PAUSE_LABEL,
        EventPropertyEnum.WORKFLOW_PAUSE_BY_PLAN,
    ),
    "pause": (
        EventEnum.WORKFLOW_PAUSE,
        EventLabelEnum.WORKFLOW_PAUSE_LABEL,
        EventPropertyEnum.WORKFLOW_PAUSE_BY_USER,
    ),
    "message": (
        EventEnum.WORKFLOW_MESSAGE,
        EventLabelEnum.WORKFLOW_MESSAGE_LABEL,
        EventPropertyEnum.WORKFLOW_MESSAGE_BY_USER,
    ),
    "resume": (
        EventEnum.WORKFLOW_RESUME,
        EventLabelEnum.WORKFLOW_RESUME_LABEL,
        EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
    ),
}


def fetch_event_mapping(
    event_type: str,
    event_by_user: bool,
) -> Tuple[EventEnum, EventLabelEnum, EventPropertyEnum]:
    """Since we cannot track an event based on event_type alone, we check if the event is by user or plan, Pause and
    Resume events can triggered by users or plans.

    We differentiate these by checking the event_by_user flag.
    """
    event_enum, label_enum, property_enum = EVENT_MAPPING[event_type]
    if event_by_user:
        return event_enum, label_enum, property_enum

    # If event by plan, Update Property accordingly
    if event_type == "pause":
        property_enum = EventPropertyEnum.WORKFLOW_PAUSE_BY_PLAN

    if event_type == "resume":
        property_enum = EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN

    return event_enum, label_enum, property_enum


@inject
def track_workflow_event(
    workflow_id: str,
    event_type: str,
    category: str,
    event_by_user: bool,
    internal_event_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
) -> None:
    """Track internal events based on the event type."""
    if event_type in EVENT_MAPPING:
        event_enum, label_enum, property_enum = fetch_event_mapping(
            event_type,
            event_by_user,
        )
        additional_properties = InternalEventAdditionalProperties(
            label=label_enum.value,
            property=property_enum.value,
            value=workflow_id,
        )
        internal_event_client.track_event(
            event_name=event_enum.value,
            additional_properties=additional_properties,
            category=category,
        )


@inject
def track_web_tool_invoked(
    tool_name: str,
    search_source: str,
    succeeded: bool,
    model_name: str,
    model_provider: str,
    internal_event_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
) -> None:
    """Count one web search or fetch, on either the native or the AgentCore path.

    Failed calls are counted too, with their status, so usage and cost reporting can decide whether to include them.
    Carries no query or URL.
    """
    monitoring_context = current_monitoring_context.get()
    internal_event_client.track_event(
        event_name=EventEnum.WEB_TOOL_INVOKED.value,
        additional_properties=InternalEventAdditionalProperties(
            label=search_source,
            property=tool_name,
            value=monitoring_context.workflow_id,
            status=(ToolStatus.SUCCESS if succeeded else ToolStatus.FAILURE).value,
        ),
        category=monitoring_context.workflow_definition or "default_category",
        model_name=model_name,
        model_provider=model_provider,
    )
