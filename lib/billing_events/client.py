import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

from lib.billing_events.context import BillingEventContext
from lib.internal_events.context import EventContext, current_event_context

__all__ = ["BillingEventsClient"]


class BillingEventsClient:
    """Client to handle billing events using SnowplowClient."""

    BILLING_CONTEXT_SCHEMA = "iglu:com.gitlab/billable_usage/jsonschema/1-0-1"

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
                batch_size=batch_size, thread_count=thread_count, endpoint=endpoint
            )

            self.snowplow_tracker = Tracker(
                app_id=app_id,
                namespace=namespace,
                emitters=[emitter],
            )

    def track_billing_event(
        self,
        event_type: str,
        category: str,
        unit_of_measure: str = "tokens",
        quantity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Send billing event to Data Insights Platform.

        Args:
            event_type: Name of billable event
            unit_of_measure: The base unit used for measurement and billing (e.g., 'bytes', 'seconds', 'tokens').
            quantity: Quantity of usage for this record.
            metadata: Dictionary containing any additional key-value pairs for billing event.
            category: The location where the billing event happened ideally classname which invoked the event.

        Returns:
            None
        """
        if not self.enabled:
            return

        if quantity <= 0:
            return

        internal_context: EventContext = current_event_context.get()

        # Unique id that will help us to map analytics events with billing events
        # This event_id will also act as an idempotent key to filter out duplicates
        event_id = str(uuid.uuid4())

        if metadata is None:
            metadata = {}

        realm_mapping = {
            "self-managed": "SM",
            "saas": "SaaS",
            "SaaS": "SaaS",
            "SM": "SM",
        }
        mapped_realm = realm_mapping.get(
            internal_context.realm or "", internal_context.realm
        )

        billing_context = BillingEventContext(
            event_id=event_id,
            event_type=event_type,
            unit_of_measure=unit_of_measure,
            quantity=quantity,
            metadata=metadata,
            subject=internal_context.global_user_id,  # TODO : We need to pass non-masked userID from GitLab instance
            global_user_id=internal_context.global_user_id,
            seat_ids=["TODO"],  # TODO : We need to pass seatIDs from GitLab instance
            realm=mapped_realm,
            timestamp=datetime.now().isoformat(),
            instance_id=internal_context.instance_id,
            unique_instance_id="",  # TODO : We need to pass unique instance_id from the GitLab instance
            host_name=internal_context.host_name,
            project_id=internal_context.project_id,
            namespace_id=internal_context.namespace_id,
            root_namespace_id=None,  # TODO : We need to pass ultimate_parent_namespace_id from the GitLab instance
            correlation_id=internal_context.correlation_id,
        )

        structured_event = StructuredEvent(
            context=[
                SelfDescribingJson(
                    self.BILLING_CONTEXT_SCHEMA, billing_context.model_dump()
                )
            ],
            category=category,
            action=event_type,
        )

        self.snowplow_tracker.track(structured_event)
