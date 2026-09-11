from dependency_injector import containers, providers

from lib.internal_events.client import InternalEventsClient
from lib.unified_events.service import UnifiedEventService

__all__ = [
    "ContainerInternalEvent",
]


class ContainerInternalEvent(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)

    client = providers.Singleton(
        InternalEventsClient,
        enabled=config.enabled,
        batch_size=config.batch_size,
        thread_count=config.thread_count,
        endpoint=config.endpoint,
        app_id=config.app_id,
        namespace=config.namespace,
    )

    unified_event_service = providers.Singleton(UnifiedEventService)
