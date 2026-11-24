from dependency_injector import containers, providers
from py_grpc_prometheus.prometheus_client_interceptor import PromClientInterceptor

from ai_gateway.integrations.container import ContainerIntegrations
from ai_gateway.models.v2.container import ContainerModels as ContainerModelsV2
from ai_gateway.prompts.container import ContainerPrompts
from ai_gateway.response_schemas.container import ContainerSchemas
from ai_gateway.searches.container import ContainerSearches
from lib.billing_events import ContainerBillingEvent
from lib.internal_events import ContainerInternalEvent
from lib.usage_quota import ContainerUsageQuota

__all__ = [
    "ContainerPlatform",
]


class ContainerPlatform(containers.DeclarativeContainer):
    """Providers that both AIGW and DWS use.

    App container classes subclass it and share its provider declarations, so injection markers that point at them
    resolve for every app. Each container instance still builds its own singletons.
    """

    config = providers.Configuration(strict=True)

    interceptor: providers.Singleton = providers.Singleton(
        PromClientInterceptor,
        enable_client_handling_time_histogram=True,
        enable_client_stream_receive_time_histogram=True,
        enable_client_stream_send_time_histogram=True,
    )

    searches = providers.Container(
        ContainerSearches,
        config=config,
    )

    internal_event = providers.Container(
        ContainerInternalEvent, config=config.internal_event
    )

    billing_event = providers.Container(
        ContainerBillingEvent,
        config=config.billing_event,
        internal_event=internal_event,
    )

    integrations = providers.Container(
        ContainerIntegrations,
        config=config,
    )

    pkg_models_v2 = providers.Container(
        ContainerModelsV2,
        config=config,
        integrations=integrations,
    )
    pkg_prompts = providers.Container(
        ContainerPrompts,
        models=pkg_models_v2,
        internal_event=internal_event,
        config=config,
    )
    pkg_schemas = providers.Container(ContainerSchemas)

    usage_quota = providers.Container(
        ContainerUsageQuota,
    )
