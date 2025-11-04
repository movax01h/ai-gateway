from dependency_injector import containers, providers

from lib.billing_events.client import BillingEventsClient
from lib.feature_flags import FeatureFlag, is_feature_enabled

__all__ = [
    "ContainerBillingEvent",
]


class ContainerBillingEvent(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    internal_event = providers.DependenciesContainer()

    client = providers.Singleton(
        BillingEventsClient,
        enabled=config.enabled
        and is_feature_enabled(FeatureFlag.DUO_USE_BILLING_ENDPOINT),
        batch_size=config.batch_size,
        thread_count=config.thread_count,
        endpoint=config.endpoint,
        app_id=config.app_id,
        namespace=config.namespace,
        internal_event_client=internal_event.client,
    )
