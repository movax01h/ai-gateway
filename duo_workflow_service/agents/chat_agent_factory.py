from typing import Optional, Union

from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.prompts import InMemoryPromptRegistry, Prompt
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.prompt_adapter import BasePromptAdapter, create_adapter
from duo_workflow_service.components.tools_registry import Toolset, ToolsRegistry
from lib.internal_events.event_enum import CategoryEnum


def create_agent(
    user: CloudConnectorUser,
    tools_registry: ToolsRegistry,
    prompt_id: str,
    prompt_version: Optional[str],
    model_metadata: Optional[TypeModelMetadata],
    internal_event_category: str,
    tools: Toolset,
    prompt_registry: Union[LocalPromptRegistry, InMemoryPromptRegistry],
    workflow_id: str,
    workflow_type: CategoryEnum,
) -> ChatAgent:
    prompt: Prompt = prompt_registry.get_on_behalf(
        user=user,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        model_metadata=model_metadata,
        internal_event_category=internal_event_category,
        tools=tools.bindable,  # type: ignore[arg-type]
        workflow_id=workflow_id,
        workflow_type=workflow_type,
    )

    # If prompt_version is None, we're using a custom agent
    use_custom_adapter = prompt_version is None
    prompt_adapter: BasePromptAdapter = create_adapter(
        prompt=prompt, use_custom_adapter=use_custom_adapter
    )

    return ChatAgent(
        name=prompt.name,
        prompt_adapter=prompt_adapter,
        tools_registry=tools_registry,
    )
