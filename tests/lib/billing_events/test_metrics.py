from itertools import product
from unittest.mock import MagicMock

import pytest
from dependency_injector import containers, providers
from prometheus_client import REGISTRY, generate_latest

from lib.billing_events import (
    BillingEvent,
    BillingEventDropReason,
    ContainerBillingEvent,
    billing_events_metrics,
)
from lib.internal_events.client import InternalEventsClient


def test_shared_instance_uses_default_registry():
    """Both AI Gateway and Duo Workflow Service expose the default registry, so the shared counters must be on it."""
    billing_events_metrics.count_emitted("code_completions")

    assert (
        REGISTRY.get_sample_value(
            "billing_events_emitted_total", {"event_type": "code_completions"}
        )
        is not None
    )


def test_container_wires_shared_instance_into_client():
    class InternalEventContainer(containers.DeclarativeContainer):
        client = providers.Object(MagicMock(spec=InternalEventsClient))

    container = ContainerBillingEvent(internal_event=InternalEventContainer())
    container.config.from_dict(
        {
            "enabled": False,
            "batch_size": 1,
            "thread_count": 1,
            "endpoint": "https://billing.local",
            "app_id": "gitlab_ai_gateway-billing",
            "namespace": "gl",
            "api_key": None,
        }
    )

    assert container.metrics() is billing_events_metrics
    assert container.client()._metrics is billing_events_metrics


_EVENT_TYPES = [event.value for event in BillingEvent]
_PAYLOAD_EVENT_TYPES = [*_EVENT_TYPES, "unknown"]


@pytest.mark.parametrize(
    "metric_name",
    [
        "billing_events_emitted_total",
        "billing_events_delivered_total",
        "billing_events_dropped_total",
    ],
)
@pytest.mark.usefixtures("metrics")
def test_counters_are_exposed_before_any_event(registry, metric_name):
    assert f"# TYPE {metric_name} counter" in generate_latest(registry).decode()


@pytest.mark.parametrize(
    ("metric_name", "labels"),
    [
        *(
            ("billing_events_emitted_total", {"event_type": event_type})
            for event_type in _EVENT_TYPES
        ),
        *(
            ("billing_events_delivered_total", {"event_type": event_type})
            for event_type in _PAYLOAD_EVENT_TYPES
        ),
        *(
            (
                "billing_events_dropped_total",
                {"reason": reason.value, "event_type": event_type},
            )
            for reason, event_type in product(
                BillingEventDropReason, _PAYLOAD_EVENT_TYPES
            )
        ),
    ],
)
@pytest.mark.usefixtures("metrics")
def test_known_series_are_zero_initialized(registry, metric_name, labels):
    """Series must exist at 0 so delivered / emitted returns data even if no event is ever delivered."""
    assert registry.get_sample_value(metric_name, labels) == 0

    exposition = generate_latest(registry).decode()
    label_text = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    assert f"{metric_name}{{{label_text}}} 0.0" in exposition


@pytest.mark.usefixtures("metrics")
def test_emitted_is_not_initialized_for_unknown_event_type(registry):
    """Emitted is always labelled with a BillingEvent value, so no unknown series is created for it."""
    assert (
        registry.get_sample_value(
            "billing_events_emitted_total", {"event_type": "unknown"}
        )
        is None
    )
