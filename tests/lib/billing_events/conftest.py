from unittest import mock
from unittest.mock import MagicMock

import pytest
from prometheus_client import CollectorRegistry

from lib.billing_events import BillingEventsClient, BillingEventsMetrics
from lib.internal_events.client import InternalEventsClient


@pytest.fixture(name="registry")
def registry_fixture():
    return CollectorRegistry()


@pytest.fixture(name="metrics")
def metrics_fixture(registry):
    return BillingEventsMetrics(registry=registry)


@pytest.fixture(name="metered_client")
def metered_client_fixture(metrics):
    """Enabled client with mocked Snowplow initializers and an isolated metrics registry."""
    with (
        mock.patch("snowplow_tracker.Tracker.__init__", return_value=None),
        mock.patch(
            "snowplow_tracker.emitters.AsyncEmitter.__init__", return_value=None
        ),
    ):
        yield BillingEventsClient(
            enabled=True,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=1,
            thread_count=1,
            internal_event_client=MagicMock(spec=InternalEventsClient),
            metrics=metrics,
        )
