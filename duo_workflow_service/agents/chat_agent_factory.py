from typing import Union

from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.prompts import InMemoryPromptRegistry, Prompt
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.prompt_adapter import DefaultPromptAdapter
from duo_workflow_service.components.tools_registry import Toolset, ToolsRegistry
from lib.internal_events.event_enum import CategoryEnum


def create_agent(
    user: CloudConnectorUser,
    tools_registry: ToolsRegistry,
    internal_event_category: str,
    tools: Toolset,
    prompt_registry: Union[LocalPromptRegistry, InMemoryPromptRegistry],
    workflow_id: str,
    workflow_type: CategoryEnum,
    system_template_override: str | None,
) -> ChatAgent:
    prompt: Prompt = prompt_registry.get_on_behalf(
        user=user,
        prompt_id="chat/agent",
        prompt_version="^1.0.0",  # type: ignore[arg-type]
        internal_event_category=internal_event_category,
        tools=tools.bindable,  # type: ignore[arg-type]
        workflow_id=workflow_id,
        workflow_type=workflow_type,
    )

    return ChatAgent(
        name=prompt.name,
        prompt_adapter=DefaultPromptAdapter(prompt),
        tools_registry=tools_registry,
        system_template_override=system_template_override,
    )
