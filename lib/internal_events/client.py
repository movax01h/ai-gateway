from dataclasses import asdict
from typing import Any, Dict, List, Optional

import structlog
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
    REQUEST_TIMEOUT = (5.0, 10.0)

    def __init__(
        self,
        enabled: bool,
        endpoint: str,
        app_id: str,
        namespace: str,
        batch_size: int,
        thread_count: int,
    ) -> None:
        self._logger = structlog.stdlib.get_logger("internal_events_client")
        self.enabled = enabled

        if enabled:
            emitter = AsyncEmitter(
                batch_size=batch_size,
                thread_count=thread_count,
                endpoint=endpoint,
                on_success=self._on_success,
                on_failure=self._on_failure,
                request_timeout=self.REQUEST_TIMEOUT,
            )

            self.snowplow_tracker = Tracker(
                app_id=app_id,
                namespace=namespace,
                emitters=[emitter],
            )

    def _on_success(self, sent_events: List[Dict[str, Any]]) -> None:
        self._logger.info(
            "Successfully sent internal events",
            sent_count=len(sent_events),
        )

        for event in sent_events:
            self._log_event(event, success=True)

    def _on_failure(
        self, succeeded_count: int, failed_events: List[Dict[str, Any]]
    ) -> None:
        self._logger.warning(
            "Failed to track internal events",
            succeeded_count=succeeded_count,
            failed_count=len(failed_events),
        )

        for event in failed_events:
            self._log_event(event, success=False)

    def _log_event(self, event_payload: Dict[str, Any], success: bool) -> None:
        log_method = self._logger.info if success else self._logger.error
        message = (
            "Internal event sent successfully"
            if success
            else "Internal event failed to send"
        )
        log_method(
            message,
            event_name=event_payload.get("se_ac"),
            label=event_payload.get("se_la"),
            property=event_payload.get("se_pr"),
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
            self._logger.debug("Internal events disabled")
            return

        if additional_properties is None:
            additional_properties = InternalEventAdditionalProperties()

        context: EventContext = current_event_context.get()
        self._logger.info("Tracking internal event", event_name=event_name)
        new_context = {
            **context.model_dump(exclude={"user_id"}),
            **kwargs,
        }
        extra = additional_properties.extra
        new_context["extra"] = extra

        session_id = additional_properties.value

        self._logger.info(
            "Building AIContext",
            event_name=event_name,
            label=additional_properties.label,
            property=additional_properties.property,
            session_id=session_id,
            workflow_id=extra.get("workflow_id"),
            flow_type=extra.get("workflow_type"),
            agent_name=extra.get("agent_name"),
            input_tokens=new_context.get("input_tokens"),
            input_tokens_type=type(new_context.get("input_tokens")).__name__,
            output_tokens=new_context.get("output_tokens"),
            output_tokens_type=type(new_context.get("output_tokens")).__name__,
            total_tokens=new_context.get("total_tokens"),
            total_tokens_type=type(new_context.get("total_tokens")).__name__,
            ephemeral_5m_input_tokens=extra.get("ephemeral_5m_input_tokens"),
            ephemeral_1h_input_tokens=extra.get("ephemeral_1h_input_tokens"),
            cache_read=extra.get("cache_read"),
        )

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
        self._logger.info(
            "Successfully called snowplow_tracker.track()",
            event_name=event_name,
        )
        tracked_internal_events.get().add(event_name)
