from dependency_injector import containers, providers

from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.chat.toolset import DuoChatToolsRegistry

__all__ = [
    "ContainerChat",
]


class ContainerChat(containers.DeclarativeContainer):
    prompts = providers.DependenciesContainer()
    internal_event = providers.DependenciesContainer()
    config = providers.Configuration(strict=True)

    _tools_registry = providers.Factory(
        DuoChatToolsRegistry,
        self_hosted_documentation_enabled=config.custom_models.enabled,
    )

    gl_agent_remote_executor_factory: providers.Factory[GLAgentRemoteExecutor] = (
        providers.Factory(
            GLAgentRemoteExecutor,
            tools_registry=_tools_registry,
            internal_event_client=internal_event.client,
        )
    )
