from dependency_injector import providers

from ai.platform.container import ContainerPlatform
from ai_gateway.auth.container import ContainerSelfSignedJwt
from ai_gateway.chat.container import ContainerChat
from ai_gateway.code_suggestions.container import ContainerCodeSuggestions
from ai_gateway.models.container import ContainerModels
from ai_gateway.tracking.container import ContainerTracking

__all__ = [
    "ContainerApplication",
]


class ContainerApplication(ContainerPlatform):
    """The AI Gateway container: the platform core plus the providers only AIGW uses."""

    snowplow = providers.Container(
        ContainerTracking, config=ContainerPlatform.config.snowplow
    )

    pkg_models = providers.Container(
        ContainerModels,
        config=ContainerPlatform.config,
        integrations=ContainerPlatform.integrations,
        internal_event=ContainerPlatform.internal_event,
        billing_event=ContainerPlatform.billing_event,
    )

    code_suggestions = providers.Container(
        ContainerCodeSuggestions,
        models=pkg_models,
        config=ContainerPlatform.config.feature_flags,
        snowplow=snowplow,
        billing_event=ContainerPlatform.billing_event,
    )
    chat = providers.Container(
        ContainerChat,
        prompts=ContainerPlatform.pkg_prompts,
        internal_event=ContainerPlatform.internal_event,
        config=ContainerPlatform.config,
    )
    self_signed_jwt = providers.Container(
        ContainerSelfSignedJwt,
        config=ContainerPlatform.config,
    )
