from dependency_injector import containers, providers

from lib.billing_events.client import BillingEventsClient

__all__ = [
    "ContainerBillingEvent",
]


class ContainerBillingEvent(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    internal_event = providers.DependenciesContainer()

    client = providers.Singleton(
        BillingEventsClient,
        enabled=config.enabled,
        batch_size=config.batch_size,
        thread_count=config.thread_count,
        endpoint=config.endpoint,
        app_id=config.app_id,
        namespace=config.namespace,
        internal_event_client=internal_event.client,
    )
