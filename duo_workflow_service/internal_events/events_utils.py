from typing import Tuple

from duo_workflow_service.internal_events import (
    DuoWorkflowInternalEvent,
    InternalEventAdditionalProperties,
)
from duo_workflow_service.internal_events.event_enum import (
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)

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


def track_workflow_event(
    workflow_id: str,
    event_type: str,
    category: str,
    event_by_user: bool,
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
        DuoWorkflowInternalEvent.track_event(
            event_name=event_enum.value,
            additional_properties=additional_properties,
            category=category,
        )
