from __future__ import annotations

import yaml
from dependency_injector.wiring import Provide, inject

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry as LocalPromptRegistry
from duo_workflow_service.agent_platform.v1.components.base import (
    ExtraInputVariablesError,
    MissingInputVariablesError,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig as V1FlowConfig,
)
from duo_workflow_service.agent_platform.v1.flows.validation import (
    _DISABLED_INTERNAL_EVENT_CLIENT,
    DryRunFlowValidator,
)
from duo_workflow_service.workflows.registry import flow_factory, get_flow_classes

# Re-export so existing callers can still import from here.
__all__ = [
    "FlowValidator",
    "MissingInputVariablesError",
    "ExtraInputVariablesError",
]


class FlowValidator:
    """Validates v1 flow configs by delegating to the production compilation path.

    ``ValidationFlow`` calls ``Flow._compile()`` with stub dependencies, exercising
    component construction, tool-name resolution, routing, and prompt-variable
    validation without touching any real external systems.

    Chat-partial flows use ``chat.Workflow`` at runtime and have empty routers by
    design; for those, only the ``flow_factory`` environment-level checks run.
    """

    @inject
    def __init__(
        self,
        prompt_registry: LocalPromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
    ) -> None:
        self._prompt_registry = prompt_registry

    def validate(self, yaml_content: str) -> None:
        """Validate a flow configuration YAML string end-to-end.

        Args:
            yaml_content: Raw YAML text of the flow configuration.

        Raises:
            ValueError: On any structural, routing, tool-name, or prompt-variable
                error found in the config.
        """
        yaml_dict = yaml.safe_load(yaml_content)
        if not isinstance(yaml_dict, dict):
            raise ValueError(
                f"Flow config must be a YAML mapping, got {type(yaml_dict).__name__}"
            )

        version = yaml_dict.get("version")
        if not version:
            raise ValueError("Missing required field 'version' in flow config")

        environment = yaml_dict.get("environment")
        if not environment:
            raise ValueError("Missing required field 'environment' in flow config")

        flow_config_cls, flow_cls = get_flow_classes(version, environment)
        config = flow_config_cls(**yaml_dict)

        # Environment-level checks: component count for chat-partial, prompt
        # security scan, etc.
        flow_factory(flow_cls, config)

        if environment == "chat-partial":
            return

        match config:
            case V1FlowConfig():
                DryRunFlowValidator(
                    config=config,
                    prompt_registry=self._prompt_registry,
                    internal_event_client=_DISABLED_INTERNAL_EVENT_CLIENT,
                ).validate()
            case _:
                raise NotImplementedError(
                    f"Dry-run validation is not implemented for config type: {type(config).__name__}"
                )
