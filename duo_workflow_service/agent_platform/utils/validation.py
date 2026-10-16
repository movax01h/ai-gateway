from __future__ import annotations

from typing import Any, Dict

import yaml
from dependency_injector.wiring import Provide, inject
from pydantic import ValidationError as PydanticValidationError

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry as LocalPromptRegistry
from duo_workflow_service.agent_platform.v1.catalog import (
    AiCatalogAgent,
    CatalogItemRef,
    CatalogItems,
    CatalogItemSource,
    CatalogItemType,
)
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
    "ExtraInputVariablesError",
    "FlowConfigLimitExceededError",
    "FlowValidationError",
    "FlowValidator",
    "MissingInputVariablesError",
]


from duo_workflow_service.agent_platform.utils.exceptions import (
    FlowConfigLimitExceededError,
    FlowValidationError,
)


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
        max_components: int | None = None,
    ) -> None:
        """Create a validator.

        Args:
            prompt_registry: Registry used to resolve prompts referenced by the flow.
            max_components: When set, configs with more entries in ``components`` are rejected before any
                Pydantic parsing or graph compilation runs, since validation cost grows with the component count.
        """
        self._prompt_registry = prompt_registry
        self._max_components = max_components

    def validate(self, yaml_content: str) -> None:
        """Validate a flow configuration YAML string end-to-end.

        Args:
            yaml_content: Raw YAML text of the flow configuration.

        Raises:
            FlowValidationError: On any validation error, including missing
                required fields, structural, routing, tool-name, or
                prompt-variable errors. The ``errors`` attribute contains
                a list of human-readable error strings.
        """
        yaml_dict = yaml.safe_load(yaml_content)
        if not isinstance(yaml_dict, dict):
            raise FlowValidationError(
                [f"Flow config must be a YAML mapping, got {type(yaml_dict).__name__}"]
            )

        self.validate_dict(yaml_dict)

    def validate_dict(self, config_dict: Dict[str, Any]) -> None:
        """Validate a flow configuration dictionary end-to-end.

        Args:
            config_dict: The flow configuration as a plain Python dict.

        Raises:
            FlowValidationError: On any validation error, including missing
                required fields, structural, routing, tool-name, or
                prompt-variable errors. The ``errors`` attribute contains
                a list of human-readable error strings.
        """
        version = config_dict.get("version")
        if not version:
            raise FlowValidationError(
                ["Missing required field 'version' in flow config"]
            )

        environment = config_dict.get("environment")
        if not environment:
            raise FlowValidationError(
                ["Missing required field 'environment' in flow config"]
            )

        self._check_component_count(config_dict)

        flow_config_cls, flow_cls = get_flow_classes(version, environment)
        try:
            config = flow_config_cls(**config_dict)
        except PydanticValidationError as exc:
            raise FlowValidationError.from_pydantic(exc) from exc

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
                    catalog_items=placeholder_items_for(config.include),
                ).validate()
            case _:
                raise NotImplementedError(
                    f"Dry-run validation is not implemented for config type: {type(config).__name__}"
                )

    def _check_component_count(self, config_dict: Dict[str, Any]) -> None:
        if self._max_components is None:
            return

        components = config_dict.get("components")
        count = len(components) if isinstance(components, list) else 0
        if count > self._max_components:
            raise FlowConfigLimitExceededError(
                [
                    f"Flow config has {count} components; the maximum is {self._max_components}"
                ]
            )


def placeholder_items_for(include: list[CatalogItemRef] | None) -> CatalogItems:
    """Return one placeholder agent for each AI Catalog entry a flow declares.

    GitLab resolves the real agents when the flow runs. The dry run still builds the supervisor and one subagent for
    each entry, so a flow that cannot build them fails validation at save.
    """
    return CatalogItems(
        ai_catalog_agents=[
            AiCatalogAgent(
                item_id=ref.item_id,
                version=ref.version or "",
                name=f"placeholder {ref.item_id}",
                description="Stands in for the agent GitLab resolves when the flow runs.",
                prompt="Stands in for the agent GitLab resolves when the flow runs.",
            )
            for ref in include or []
            if ref.source == CatalogItemSource.AI_CATALOG
            and ref.item_type == CatalogItemType.AGENT
            and ref.item_id.strip()
            and (ref.version or "").strip()
        ]
    )
