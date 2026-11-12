from dependency_injector import containers, providers

from lib.billing_events.client import BillingEventsClient
from lib.billing_events.metrics import billing_events_metrics
from lib.billing_events.service import BillingEventService

__all__ = [
    "ContainerBillingEvent",
]


class ContainerBillingEvent(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    internal_event = providers.DependenciesContainer()

    # Shared instance on the default registry, so AI Gateway and Duo Workflow Service both expose the counters and
    # rebuilding the container does not register them twice.
    metrics = providers.Object(billing_events_metrics)

    client = providers.Singleton(
        BillingEventsClient,
        enabled=config.enabled,
        batch_size=config.batch_size,
        thread_count=config.thread_count,
        endpoint=config.endpoint,
        app_id=config.app_id,
        namespace=config.namespace,
        internal_event_client=internal_event.client,
        api_key=config.api_key,
        metrics=metrics,
    )

    service = providers.Factory(
        BillingEventService,
        client=client,
    )
