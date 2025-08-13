from dependency_injector import containers, providers

from ai_gateway.chat import agents as chat
from ai_gateway.config import ConfigModelLimits
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.registry import LocalPromptRegistry

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
            "chat/agent": "duo_workflow_service.agents.chat_agent.ChatAgent",
            "workflow/convert_to_gitlab_ci": "duo_workflow_service.agents.Agent",
            "workflow/executor": "duo_workflow_service.agents.Agent",
            "workflow/context_builder": "duo_workflow_service.agents.Agent",
            "workflow/planner": "duo_workflow_service.agents.Agent",
            "workflow/goal_disambiguation": "duo_workflow_service.agents.Agent",
            "workflow/issue_to_merge_request": "duo_workflow_service.agents.Agent",
        },
        model_factories={
            ModelClassProvider.ANTHROPIC: providers.Factory(
                models.anthropic_claude_chat_fn
            ),
            ModelClassProvider.OPENAI: providers.Factory(
                models.openai_chat_fn, verbosity="low", reasoning={"effort": "low"}
            ),
            ModelClassProvider.LITE_LLM: providers.Factory(models.lite_llm_chat_fn),
            ModelClassProvider.AMAZON_Q: providers.Factory(models.amazon_q_chat_fn),
        },
        internal_event_client=internal_event.client,
        model_limits=providers.Factory(ConfigModelLimits, config.model_engine_limits),
        custom_models_enabled=config.custom_models.enabled,
        disable_streaming=config.custom_models.disable_streaming,
    )
