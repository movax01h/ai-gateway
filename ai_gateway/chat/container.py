from dependency_injector import containers, providers

from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.chat.toolset import DuoChatToolsRegistry

__all__ = [
    "ContainerChat",
]


class ContainerChat(containers.DeclarativeContainer):
    prompts = providers.DependenciesContainer()
    models = providers.DependenciesContainer()
    internal_event = providers.DependenciesContainer()
    config = providers.Configuration(strict=True)

    # The dependency injector does not allow us to override the FactoryAggregate provider directly.
    # However, we can still override its internal sub-factories to achieve the same goal.
    _anthropic_claude_llm_factory = providers.Factory(models.anthropic_claude)
    _anthropic_claude_chat_factory = providers.Factory(models.anthropic_claude_chat)

    # We need to resolve the model based on model name provided in request payload
    # Hence, `models._anthropic_claude` and `models._anthropic_claude_chat_factory` are only partially applied here.
    anthropic_claude_factory = providers.FactoryAggregate(
        llm=_anthropic_claude_llm_factory, chat=_anthropic_claude_chat_factory
    )

    litellm_factory = providers.Factory(models.litellm_chat)

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
