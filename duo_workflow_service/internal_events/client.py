import os
import threading
from typing import Optional

from pydantic import BaseModel
from snowplow_tracker.emitters import AsyncEmitter
from snowplow_tracker.events.structured_event import StructuredEvent
from snowplow_tracker.self_describing_json import SelfDescribingJson
from snowplow_tracker.tracker import Tracker

from duo_workflow_service.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)
from duo_workflow_service.tracking.errors import log_exception

__all__ = ["InternalEventsClient", "InternalEventsConfig", "DuoWorkflowInternalEvent"]

SNOWPLOW_EVENT_NAMESPACE = "gl"


class InternalEventsConfig(BaseModel):
    enabled: bool
    endpoint: str
    app_id: str
    namespace: str
    batch_size: int
    thread_count: int


class InternalEventsClient:
    """Client to handle internal events using SnowplowClient."""

    STANDARD_CONTEXT_SCHEMA = "iglu:com.gitlab/gitlab_standard/jsonschema/1-1-1"

    def __init__(self, config: InternalEventsConfig) -> None:
        self.enabled = config.enabled

        if config.enabled:
            emitter = AsyncEmitter(
                batch_size=config.batch_size,
                thread_count=config.thread_count,
                endpoint=config.endpoint,
            )

            self.snowplow_tracker = Tracker(
                app_id=config.app_id,
                namespace=config.namespace,
                emitters=[emitter],
            )

    def track_event(
        self,
        event_name: str,
        additional_properties: Optional[InternalEventAdditionalProperties] = None,
        category: str = "default_category",
    ) -> None:
        """Send internal event to Snowplow.

        Args:
            event_name: The name of the event. It should follow
                <action>_<target_of_action>_<where/when>. Reference:
                https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html#defining-event-and-metrics
            additional_properties: Additional properties for the event.
            category: the location where the event happened ideally the name of the class which invoked the event.
        """
        try:
            if not self.enabled:
                return

            if additional_properties is None:
                additional_properties = InternalEventAdditionalProperties()

            context: EventContext = current_event_context.get()
            new_context = context.model_dump()
            new_context["extra"] = additional_properties.extra

            structured_event = StructuredEvent(
                context=[SelfDescribingJson(self.STANDARD_CONTEXT_SCHEMA, new_context)],
                category=category,
                action=event_name,
                label=additional_properties.label,
                value=additional_properties.value,
                property_=additional_properties.property,
            )

            self.snowplow_tracker.track(structured_event)
        except Exception as e:
            log_exception(e, {"message": "Failed to send internal tracking event"})


class DuoWorkflowInternalEvent:
    __singleton_lock = threading.Lock()
    __singleton_instance = None

    @classmethod
    def setup(cls):
        try:
            if not cls.__singleton_instance:
                with cls.__singleton_lock:
                    config = InternalEventsConfig(
                        enabled=os.getenv("DW_INTERNAL_EVENT__ENABLED", "false").lower()
                        == "true",
                        endpoint=os.getenv("DW_INTERNAL_EVENT__ENDPOINT", ""),
                        app_id=os.getenv("DW_INTERNAL_EVENT__APP_ID", ""),
                        namespace=SNOWPLOW_EVENT_NAMESPACE,
                        batch_size=int(os.getenv("DW_INTERNAL_EVENT__BATCH_SIZE", "1")),
                        thread_count=int(
                            os.getenv("DW_INTERNAL_EVENT__THREAD_COUNT", "1")
                        ),
                    )
                    cls.__singleton_instance = InternalEventsClient(config)
        except Exception as e:
            log_exception(e, {"message": "Failed to set up internal events tracking"})

    @classmethod
    def track_event(
        cls,
        event_name: str,
        additional_properties: Optional[InternalEventAdditionalProperties] = None,
        category: str = "default_category",
    ) -> None:
        if cls.__singleton_instance:
            cls.__singleton_instance.track_event(
                event_name, additional_properties, category
            )

    @classmethod
    def instance(cls):
        return cls.__singleton_instance
