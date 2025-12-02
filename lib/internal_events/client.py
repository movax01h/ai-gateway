from dataclasses import asdict
from typing import Optional

from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

from lib.internal_events.ai_context import AIContext
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
    tracked_internal_events,
)

__all__ = ["InternalEventsClient"]


class InternalEventsClient:
    """Client to handle internal events using SnowplowClient."""

    STANDARD_CONTEXT_SCHEMA = "iglu:com.gitlab/gitlab_standard/jsonschema/1-1-7"
    AI_CONTEXT_SCHEMA = "iglu:com.gitlab/ai_context/jsonschema/1-0-0"

    def __init__(
        self,
        enabled: bool,
        endpoint: str,
        app_id: str,
        namespace: str,
        batch_size: int,
        thread_count: int,
    ) -> None:
        self.enabled = enabled

        if enabled:
            emitter = AsyncEmitter(
                batch_size=batch_size,
                thread_count=thread_count,
                endpoint=endpoint,
            )

            self.snowplow_tracker = Tracker(
                app_id=app_id,
                namespace=namespace,
                emitters=[emitter],
            )

    def track_event(
        self,
        event_name: str,
        additional_properties: Optional[InternalEventAdditionalProperties] = None,
        category: Optional[str] = "default_category",
        **kwargs,
    ) -> None:
        """Send internal event to Snowplow.

        Args:
            event_name: The name of the event. It should follow
                <action>_<target_of_action>_<where/when>. Reference:
                https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#defining-event-and-metrics
            additional_properties: Additional properties for the event.
            category:  the location where the event happened ideally classname which invoked the event.
        """
        if not self.enabled:
            return

        if additional_properties is None:
            additional_properties = InternalEventAdditionalProperties()

        context: EventContext = current_event_context.get()
        new_context = {
            **context.model_dump(exclude={"user_id"}),
            **kwargs,
        }
        extra = additional_properties.extra
        new_context["extra"] = extra

        session_id = additional_properties.value

        ai_context = AIContext(
            session_id=str(session_id) if session_id is not None else None,
            workflow_id=extra.get("workflow_id"),
            flow_type=extra.get("workflow_type"),
            agent_name=extra.get("agent_name"),
            input_tokens=new_context.get("input_tokens"),
            output_tokens=new_context.get("output_tokens"),
            total_tokens=new_context.get("total_tokens"),
            ephemeral_5m_input_tokens=extra.get("ephemeral_5m_input_tokens"),
            ephemeral_1h_input_tokens=extra.get("ephemeral_1h_input_tokens"),
            cache_read=extra.get("cache_read"),
        )

        structured_event = StructuredEvent(
            context=[
                SelfDescribingJson(self.STANDARD_CONTEXT_SCHEMA, new_context),
                SelfDescribingJson(self.AI_CONTEXT_SCHEMA, asdict(ai_context)),
            ],
            category=category,  # type: ignore[arg-type]
            action=event_name,
            label=additional_properties.label,
            value=additional_properties.value,
            property_=additional_properties.property,
        )

        self.snowplow_tracker.track(structured_event)
        tracked_internal_events.get().add(event_name)
