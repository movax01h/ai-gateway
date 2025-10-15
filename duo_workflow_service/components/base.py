from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.llm_factory import AnthropicConfig, VertexConfig
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.internal_events.event_enum import CategoryEnum


class BaseComponent:
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
        self.user = user
        self.prompt_registry = prompt_registry
