from typing import Union

from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.prompts import InMemoryPromptRegistry, Prompt
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.prompt_adapter import DefaultPromptAdapter
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.components.tools_registry import Toolset, ToolsRegistry
from lib.events import GLReportingEventContext
from lib.feature_flags.context import FeatureFlag, is_feature_enabled


def create_agent(
    user: CloudConnectorUser,
    tools_registry: ToolsRegistry,
    internal_event_category: str,
    tools: Toolset,
    prompt_registry: Union[LocalPromptRegistry, InMemoryPromptRegistry],
    workflow_id: str,
    workflow_type: GLReportingEventContext,
    system_template_override: str | None,
    agent_name_override: str | None = None,
) -> ChatAgent:
    # Use agent_name_override for chat-partial flows, default to "chat"
    agent_name = agent_name_override if agent_name_override else "chat"

    # Include web_search_options conditionally based on feature flag and client capability
    bind_tools_params: dict[str, dict] = {}
    if is_feature_enabled(FeatureFlag.DAP_WEB_SEARCH) and is_client_capable(
        "web_search"
    ):
        bind_tools_params["web_search_options"] = {}

    prompt: Prompt = prompt_registry.get_on_behalf(
        user=user,
        prompt_id="chat/agent",
        prompt_version="^1.0.0",
        internal_event_category=internal_event_category,
        tools=tools.bindable,  # type: ignore[arg-type]
        bind_tools_params=bind_tools_params,
        internal_event_extra={
            "agent_name": agent_name,
            "workflow_id": workflow_id,
            "workflow_type": workflow_type.value,
        },
    )

    return ChatAgent(
        name=prompt.name,
        prompt_adapter=DefaultPromptAdapter(prompt),
        tools_registry=tools_registry,
        system_template_override=system_template_override,
    )
