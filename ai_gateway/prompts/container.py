from dependency_injector import containers, providers

from ai_gateway.chat import agents as chat
from ai_gateway.config import ConfigModelLimits
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent

__all__ = [
    "ContainerPrompts",
]


class ContainerPrompts(containers.DeclarativeContainer):
    config = providers.Configuration(strict=True)
    models = providers.DependenciesContainer()
    internal_event = providers.DependenciesContainer()

    prompt_registry = providers.Singleton(
        LocalPromptRegistry.from_local_yaml,
        class_overrides={
            "chat/react": chat.ReActAgent,
            "chat/react/vertex": chat.ReActAgent,
            "chat/agent": ChatAgent,
        },
        model_factories={
            ModelClassProvider.ANTHROPIC: providers.Factory(
                models.anthropic_claude_chat_fn
            ),
            ModelClassProvider.LITE_LLM: providers.Factory(models.lite_llm_chat_fn),
            ModelClassProvider.AMAZON_Q: providers.Factory(models.amazon_q_chat_fn),
        },
        default_prompts=config.default_prompts,
        internal_event_client=internal_event.client,
        model_limits=providers.Factory(ConfigModelLimits, config.model_engine_limits),
        custom_models_enabled=config.custom_models.enabled,
        disable_streaming=config.custom_models.disable_streaming,
    )
