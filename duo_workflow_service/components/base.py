import json
from typing import Dict, Type

from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from pydantic import BaseModel, ValidationError

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.llm_factory import AnthropicConfig, VertexConfig
from duo_workflow_service.workflows.type_definitions import (
    AdditionalContext,
    OsInformationContext,
)
from lib.internal_events.event_enum import CategoryEnum

_CONTEXT_REGISTRY: Dict[Type[BaseModel], str] = {
    OsInformationContext: "os_information_context",
}

_TYPE_SIGNATURES: Dict[frozenset, Type[BaseModel]] = {
    frozenset(context_type.model_fields.keys()): context_type
    for context_type in _CONTEXT_REGISTRY
}


class BaseComponent:  # pylint: disable=too-many-instance-attributes; there'll be less as we migrate to Prompt Registry
    @inject
    def __init__(
        self,
        workflow_id: str,
        workflow_type: CategoryEnum,
        goal: str,
        tools_registry: ToolsRegistry,
        model_config: AnthropicConfig | VertexConfig,
        http_client: GitlabHttpClient,
        additional_context: list[AdditionalContext] | None = None,
        user: CloudConnectorUser | None = None,
        prompt_registry: LocalPromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
    ):
        self.model_config = model_config
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.goal = goal
        self.tools_registry = tools_registry
        self.http_client = http_client
        self.additional_context = additional_context
        self.agent_user_environment = _process_agent_user_environment(
            additional_context
        )
        self.user = user
        self.prompt_registry = prompt_registry


def _process_agent_user_environment(
    additional_contexts: list[AdditionalContext] | None = None,
) -> Dict[str, BaseModel]:
    """Process and assign contexts to appropriate fields."""

    if additional_contexts is None or len(additional_contexts) == 0:
        return {}

    contexts = {}

    for context in additional_contexts:
        if context.category != "agent_user_environment" or not context.content:
            continue

        try:
            data = json.loads(context.content)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        # Check which type matches
        data_fields = frozenset(data.keys())

        if data_fields in _TYPE_SIGNATURES:
            context_type = _TYPE_SIGNATURES[data_fields]
            field_name = _CONTEXT_REGISTRY[context_type]

            try:
                instance = context_type.model_validate(data)
                contexts[field_name] = instance
            except ValidationError:
                continue

    return contexts
