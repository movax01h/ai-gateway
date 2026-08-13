"""Product-analytics events for human-in-the-loop tool governance.

Privacy: payloads carry only the tool name, flow type, workflow id, and
outcome - never tool arguments, user/rejection text, patterns, or commands.
"""

from typing import Optional

from duo_workflow_service.tracking.errors import log_exception
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum

__all__ = ["track_tool_governance_event"]


def track_tool_governance_event(
    internal_event_client: InternalEventsClient,
    event_name: EventEnum,
    *,
    flow_type: str,
    flow_id: str,
    tool_name: Optional[str] = None,
    outcome: Optional[str] = None,
) -> None:
    """Track a tool-governance internal event.

    Tracking is strictly best-effort: any exception raised while building or
    emitting the event is logged and swallowed, so tracking must never break
    the enforcement paths (tool responses, approval flow) that call it.

    Args:
        internal_event_client: Client the event is emitted through.
        event_name: Governance event to emit.
        flow_type: Flow/workflow type (e.g. ``chat``), also used as category.
        flow_id: Workflow id used to correlate with other workflow events.
        tool_name: Name of the tool involved, when known.
        outcome: Approval outcome (``approval``/``rejection``/``modification``)
            for resolution events; ``None`` otherwise.
    """
    try:
        extra: dict = {}
        if tool_name is not None:
            extra["tool_name"] = tool_name

        additional_properties = InternalEventAdditionalProperties(
            label=flow_type,
            property=outcome or tool_name,
            value=flow_id,
            **extra,
        )
        internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=flow_type,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log_exception(e, extra={"event_name": str(event_name)})
