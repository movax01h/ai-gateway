import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests
import structlog
from gitlab_cloud_connector import CloudConnectorUser
from pydantic import SecretStr, ValidationError
from snowplow_tracker import AsyncEmitter, SelfDescribingJson, StructuredEvent, Tracker

from lib.billing_events.context import BillingEventContext, normalize_realm
from lib.billing_events.events import BillingEvent
from lib.billing_events.metrics import (
    KNOWN_EVENT_TYPES,
    UNKNOWN_EVENT_TYPE,
    BillingEventDropReason,
    BillingEventsMetrics,
    billing_events_metrics,
)
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)
from lib.snowplow import set_bearer_token
from lib.usage_quota.client import should_skip_usage_quota_for_user

__all__ = ["BillingEvent", "BillingEventsClient"]


def _event_type_label(payload: Mapping[str, Any]) -> str:
    """Return the billing event type carried by a Snowplow payload, for use as a metric label.

    The structured event action (``se_ac``) is set to the ``BillingEvent`` value at enqueue time, so it is available on
    emitter worker threads without reading request context. Unexpected values map to ``unknown`` to bound cardinality.
    """
    value = payload.get("se_ac")
    return value if value in KNOWN_EVENT_TYPES else UNKNOWN_EVENT_TYPE


class _BillingAsyncEmitter(AsyncEmitter):
    """AsyncEmitter that can tell whether the last failed request on the current thread will be retried.

    Snowplow calls ``on_failure`` after every failed attempt, before it decides whether to retry, and does not pass the
    HTTP status. ``send_events`` runs ``http_post`` and then the callbacks on the same emitter worker thread, so the
    status is recorded per thread.

    ``last_failure_is_retryable`` relies on the private ``Emitter._should_retry``, so that it follows Snowplow's own retry
    rules (including ``custom_retry_codes``). Re-check it when upgrading ``snowplow-tracker``.
    """

    _last_request = threading.local()

    def http_post(self, data: str) -> int:
        status_code = super().http_post(data)
        self._last_request.status_code = status_code
        return status_code

    def last_failure_is_retryable(self) -> bool:
        # http_post returns -1 when the request raises (for example a connection error); Snowplow retries those.
        return self._should_retry(getattr(self._last_request, "status_code", -1))


class BillingEventsClient:
    """Client to handle billing events using SnowplowClient."""

    BILLING_CONTEXT_SCHEMA = "iglu:com.gitlab/billable_usage/jsonschema/1-0-3"

    def __init__(
        self,
        enabled: bool,
        endpoint: str,
        app_id: str,
        namespace: str,
        batch_size: int,
        thread_count: int,
        internal_event_client: InternalEventsClient,
        api_key: Optional[SecretStr] = None,
        metrics: BillingEventsMetrics = billing_events_metrics,
    ) -> None:
        self._logger = structlog.stdlib.get_logger("billing_events_client")
        self.enabled = enabled
        self.internal_event_client = internal_event_client
        self._metrics = metrics

        self._logger.info(
            "Initializing BillingEventsClient",
            enabled=enabled,
            endpoint=endpoint,
            app_id=app_id,
            namespace=namespace,
            batch_size=batch_size,
            thread_count=thread_count,
            api_key_configured=bool(api_key),
        )

        if enabled:
            self._logger.info("Creating AsyncEmitter and Tracker for billing events")
            self._session = requests.Session()
            set_bearer_token(self._session, api_key)
            self._emitter = _BillingAsyncEmitter(
                batch_size=batch_size,
                thread_count=thread_count,
                endpoint=endpoint,
                on_success=self._on_success,
                on_failure=self._on_failure,
                session=self._session,
            )

            self.snowplow_tracker = Tracker(
                app_id=app_id,
                namespace=namespace,
                emitters=[self._emitter],
            )
            self._logger.info("Successfully initialized billing events tracker")
        else:
            self._logger.info(
                "Billing events disabled - skipping tracker initialization"
            )

    def _get_emitter_stats(self) -> Tuple[int, int]:
        if not self.enabled:
            return -1, -1
        try:
            queue_size = self._emitter.queue.qsize()
        except AttributeError:
            queue_size = -1
        try:
            buffer_size = self._emitter.event_store.size()
        except AttributeError:
            buffer_size = -1
        return queue_size, buffer_size

    def shutdown(self) -> None:
        if not self.enabled:
            return

        queue_size, buffer_size = self._get_emitter_stats()
        self._logger.info(
            "Shutting down billing events client",
            emitter_queue_size=queue_size,
            emitter_buffer_size=buffer_size,
            emitter_total_pending=(
                queue_size + buffer_size if queue_size >= 0 and buffer_size >= 0 else -1
            ),
        )
        self.snowplow_tracker.flush(is_async=True)
        self._session.close()

    def _on_success(self, sent_events: List[Dict[str, Any]]) -> None:
        self._logger.info(
            "Successfully sent billing events",
            sent_count=len(sent_events),
        )

        for event in sent_events:
            self._log_event(event, success=True)
            self._metrics.count_delivered(_event_type_label(event))

    def _on_failure(
        self, succeeded_count: int, failed_events: List[Dict[str, Any]]
    ) -> None:
        # Runs on an emitter worker thread: labels come from the event payloads, not from request context.
        will_retry = self._emitter.last_failure_is_retryable()
        self._logger.warning(
            "Failed to track billing events",
            succeeded_count=succeeded_count,
            failed_count=len(failed_events),
            will_retry=will_retry,
        )

        for event in failed_events:
            self._log_event(event, success=False)
            # Retryable failures go back into the buffer and are not lost yet, so they are not counted as dropped. They
            # stay in the emitted - delivered gap until a retry succeeds.
            if not will_retry:
                self._metrics.count_dropped(
                    BillingEventDropReason.HTTP_ERROR, _event_type_label(event)
                )

    def _log_event(self, event_payload: Dict[str, Any], success: bool) -> None:
        log_method = self._logger.info if success else self._logger.error
        message = (
            "Billing event sent successfully"
            if success
            else "Billing event failed to send"
        )
        log_method(
            message,
            event_name=event_payload.get("se_ac"),
            event_id=event_payload.get("eid"),
            label=event_payload.get("se_la"),
            property=event_payload.get("se_pr"),
        )

    def record_emitted(self, event: BillingEvent) -> None:
        """Count one billing attempt in ``billing_events_emitted_total`` while billing events are enabled.

        Call this exactly once per billing attempt, before anything that can fail, then hand the event to
        ``send_billing_event``. ``track_billing_event`` does both.
        """
        if self.enabled:
            self._metrics.count_emitted(event.value)

    def record_dropped(
        self, reason: BillingEventDropReason, event: BillingEvent
    ) -> None:
        """Count a billing event dropped before it reached the Snowplow tracker, while billing events are enabled."""
        if self.enabled:
            self._metrics.count_dropped(reason, event.value)

    def track_billing_event(
        self,
        user: CloudConnectorUser,
        event: BillingEvent,
        category: str,
        unit_of_measure: str = "tokens",
        quantity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Count a billing attempt as emitted, then send it to Data Insights Platform.

        Entry point for callers that use the client directly. ``BillingEventService.track_billing`` counts the attempt
        itself and calls ``send_billing_event``, so each attempt is counted as emitted exactly once on either path.

        Args and return value are the same as ``send_billing_event``.
        """
        self.record_emitted(event)
        return self.send_billing_event(
            user,
            event,
            category,
            unit_of_measure=unit_of_measure,
            quantity=quantity,
            metadata=metadata,
        )

    def send_billing_event(
        self,
        user: CloudConnectorUser,
        event: BillingEvent,
        category: str,
        unit_of_measure: str = "tokens",
        quantity: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send billing event to Data Insights Platform, without counting it as emitted.

        The caller must already have counted the attempt with ``record_emitted``. Use ``track_billing_event`` unless
        you own that count, as ``BillingEventService.track_billing`` does.

        Args:
            user: CloudConnectorUser object containing user claims and authentication information.
            event: BillingEvent enum value representing the type of billable event (e.g., CODE_COMPLETIONS,
                AIGW_PROXY_USE).
            category: The location where the billing event occurred, ideally the class name which invoked the event.
            unit_of_measure: The base unit used for measurement and billing (e.g., 'bytes', 'seconds', 'tokens',
                'request'). Defaults to 'tokens'.
            quantity: Quantity of usage for this record. Must be greater than 0. Defaults to 1.0.
            metadata: Optional dictionary containing additional key-value pairs for the billing event context.

        Returns:
            True if the event was handed to the Snowplow tracker, False if it was skipped (billing events disabled or
            invalid quantity) or the Snowplow call failed. A failure in the follow-up internal analytics event does not
            change the result, because the billing event has already been sent by then.
        """
        if not self.enabled:
            self._logger.info("Billing events disabled")
            return False

        self._logger.info("Tracking billing event", event_type=event.value)

        if quantity <= 0:
            self._logger.warning("Invalid quantity", quantity=quantity)
            self.record_dropped(BillingEventDropReason.MISSING_FIELD, event)
            return False

        internal_context: EventContext = current_event_context.get()

        # Unique id that will help us to map analytics events with billing events
        # This event_id will also act as an idempotent key to filter out duplicates
        event_id = str(uuid.uuid4())

        metadata = {
            **(metadata or {}),
            "is_gitlab_team_member": should_skip_usage_quota_for_user(user),
        }

        mapped_realm = normalize_realm(internal_context.realm)
        unique_instance_id = user.claims.gitlab_instance_uid if user.claims else None

        # Pass the feature_enablement_type (if present) as a list. This could be expanded in the future.
        # See https://gitlab.com/gitlab-org/architecture/usage-billing/design-doc/-/issues/20#note_2727864683
        assignments = (
            [internal_context.feature_enablement_type]
            if internal_context.feature_enablement_type
            else None
        )

        try:
            billing_context = BillingEventContext(
                event_id=event_id,
                event_type=event.value,
                unit_of_measure=unit_of_measure,
                quantity=quantity,
                metadata=metadata,
                subject=internal_context.user_id,
                global_user_id=internal_context.global_user_id,
                realm=mapped_realm,
                timestamp=datetime.now().isoformat(),
                instance_id=internal_context.instance_id,
                unique_instance_id=unique_instance_id,
                host_name=internal_context.host_name,
                project_id=internal_context.project_id,
                namespace_id=internal_context.namespace_id,
                root_namespace_id=internal_context.ultimate_parent_namespace_id,
                correlation_id=internal_context.correlation_id,
                deployment_type=internal_context.deployment_type,
                subject_type=internal_context.user_type,
                assignments=assignments,
                instance_version=internal_context.instance_version,
                organization_id=internal_context.organization_id,
            )
        except ValidationError:
            # Count the rejected event, then re-raise so callers see the same error as before.
            self.record_dropped(BillingEventDropReason.MISSING_FIELD, event)
            raise

        structured_event = StructuredEvent(
            context=[
                SelfDescribingJson(
                    self.BILLING_CONTEXT_SCHEMA, billing_context.model_dump()
                )
            ],
            category=category,
            action=event.value,
        )

        try:
            self.snowplow_tracker.track(structured_event)
            self._logger.info(
                "Successfully called snowplow_tracker.track()",
                event_id=event_id,
            )
        except Exception as e:
            self._logger.error(
                "Failed to send billing event",
                error=str(e),
            )
            return False

        # The billing event has been sent at this point. The internal analytics event is best-effort and must not
        # affect the result, otherwise a caller retrying on False could bill twice.
        try:
            additional_properties = InternalEventAdditionalProperties(
                label=event_id,
                property=event.value,
                metadata=metadata,
            )
            self.internal_event_client.track_event(
                event_name="usage_billing_event",
                additional_properties=additional_properties,
                category=category,
            )
            self._logger.info(
                "Successfully called internal_event_client.track_event()",
                event_id=event_id,
            )
        except Exception as e:
            self._logger.error(
                "Failed to send internal event for billing event",
                event_id=event_id,
                error=str(e),
            )

        return True
