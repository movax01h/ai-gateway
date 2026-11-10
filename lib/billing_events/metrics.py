"""Prometheus metrics for billing event delivery.

Billing events are emitted by both the AI Gateway and the Duo Workflow Service, so these counters live in shared
``lib/`` code and register against the default ``prometheus_client`` registry, which both services expose on their
metrics endpoints.

The reconciliation ratio is ``billing_events_delivered_total / billing_events_emitted_total``. Both sides of the ratio
carry only the ``event_type`` label: delivery is reported from Snowplow emitter worker threads, where request-scoped
context variables (and therefore the shared ``<metadata>`` labels) resolve to ``unknown``. ``event_type`` is set at
enqueue time and travels with the event payload, so it is identical on both sides.

Every known label combination is initialized at 0 when the counters are created. Prometheus series otherwise only appear
on their first increment, so if the billing endpoint were down from the moment a pod starts,
``billing_events_delivered_total`` would not exist and the ratio would return no data instead of 0.
"""

from enum import StrEnum

from prometheus_client import REGISTRY, CollectorRegistry, Counter

from lib.billing_events.events import BillingEvent

__all__ = [
    "BillingEventDropReason",
    "BillingEventsMetrics",
    "billing_events_metrics",
]

UNKNOWN_EVENT_TYPE = "unknown"

KNOWN_EVENT_TYPES: frozenset[str] = frozenset(event.value for event in BillingEvent)


class BillingEventDropReason(StrEnum):
    """Why a billing event was dropped before delivery.

    Keep this set bounded: it is used as a Prometheus label value.
    """

    # A required field is missing or invalid, so the event was rejected before it was handed to Snowplow.
    MISSING_FIELD = "missing_field"
    # The billing endpoint returned a status that Snowplow does not retry (for example 400, 401, 403, 410, 422).
    HTTP_ERROR = "http_error"


class BillingEventsMetrics:
    """Counters for billing events emitted, delivered, and dropped."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.emitted_counter = Counter(
            "billing_events_emitted_total",
            "Count of billing events expected to be delivered (handed to the billing client while enabled)",
            ["event_type"],
            registry=registry,
        )
        self.delivered_counter = Counter(
            "billing_events_delivered_total",
            "Count of billing events successfully delivered to the billing endpoint",
            ["event_type"],
            registry=registry,
        )
        self.dropped_counter = Counter(
            "billing_events_dropped_total",
            "Count of billing events dropped before delivery",
            ["reason", "event_type"],
            registry=registry,
        )
        self._initialize_series()

    def _initialize_series(self) -> None:
        """Create every known series at 0 so the reconciliation ratio returns data before the first event.

        ``emitted`` is always labelled with a ``BillingEvent`` value. ``delivered`` and ``dropped`` read the label back
        from the event payload and fall back to ``unknown``, so that value is initialized for them too. The label sets
        are bounded: ``len(BillingEvent) + 1`` event types and ``len(BillingEventDropReason)`` reasons.
        """
        payload_event_types = sorted(KNOWN_EVENT_TYPES) + [UNKNOWN_EVENT_TYPE]

        for event_type in sorted(KNOWN_EVENT_TYPES):
            self.emitted_counter.labels(event_type=event_type)

        for event_type in payload_event_types:
            self.delivered_counter.labels(event_type=event_type)
            for reason in BillingEventDropReason:
                self.dropped_counter.labels(reason=reason.value, event_type=event_type)

    def count_emitted(self, event_type: str) -> None:
        """Increment the emitted counter for one billing event.

        Args:
            event_type: Billing event type (a ``BillingEvent`` value).
        """
        self.emitted_counter.labels(event_type=event_type).inc()

    def count_delivered(self, event_type: str, amount: int = 1) -> None:
        """Increment the delivered counter.

        Args:
            event_type: Billing event type (a ``BillingEvent`` value), or ``unknown``.
            amount: Number of events delivered.
        """
        self.delivered_counter.labels(event_type=event_type).inc(amount)

    def count_dropped(
        self,
        reason: BillingEventDropReason,
        event_type: str = UNKNOWN_EVENT_TYPE,
        amount: int = 1,
    ) -> None:
        """Increment the dropped counter.

        Args:
            reason: Why the events were dropped.
            event_type: Billing event type (a ``BillingEvent`` value), or ``unknown``.
            amount: Number of events dropped.
        """
        self.dropped_counter.labels(reason=reason.value, event_type=event_type).inc(
            amount
        )


# Shared instance registered against the default registry. Import this rather than instantiating
# BillingEventsMetrics again: a second instance raises a duplicate-registration error.
billing_events_metrics = BillingEventsMetrics()
