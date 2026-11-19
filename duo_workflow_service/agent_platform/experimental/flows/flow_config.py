from pathlib import Path
from typing import Any, Callable, ClassVar, List, Literal, Optional, override

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from ai_gateway.response_schemas.config import InlineResponseSchemaConfig
from duo_workflow_service.agent_platform.experimental.components import (
    BaseComponent,
    ComponentRegistry,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    MCP_AUTO_INJECT_ENVIRONMENTS,
    BaseFlowConfig,
    ChatPartialMixin,
    FlowConfigInput,
    FlowConfigMetadata,
    list_flow_configs,
    load_registry_config,
    load_registry_file,
    select_config_class,
)

__all__ = [
    "CONFIG_CLASSES",
    "FlowConfig",
    "FlowConfigInput",
    "FlowConfigMetadata",
    "PartialFlowConfig",
    "config_from_data",
    "list_configs",
    "load_component_class",
    "load_config",
    "load_config_file",
]

_EXPERIMENTAL_MCP_AUTO_INJECT_ENVIRONMENTS = MCP_AUTO_INJECT_ENVIRONMENTS | frozenset(
    ("ide",)
)


class FlowConfig(BaseFlowConfig):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"
    FEATURE_CONFIG_SUBDIR: ClassVar[Optional[str]] = "experimental"
    # Validates that a file under ``config/experimental/`` declares its tier.
    version: Literal["experimental"]
    prompts: Optional[list[InMemoryPromptConfig]] = None
    response_schemas: Optional[list[InlineResponseSchemaConfig]] = None
    # A registry flow that cannot work without its ``for_each`` fan-outs runs them
    # whether or not the ``dap_for_each`` feature flag is on.
    requires_for_each: bool = False

    def should_auto_inject_mcp_tools(self) -> bool:
        """Return whether MCP tools should be automatically injected into this flow's toolset.

        Extends the base behaviour by also enabling MCP auto-injection for ``ide``
        environment flows, which support human-in-the-loop interactions via
        :class:`HumanInputComponent` and require access to session-connected MCP tools.

        Returns:
            ``True`` when the flow's ``environment`` is ``"chat"``, ``"chat-partial"``,
            or ``"ide"``; ``False`` otherwise.
        """
        return self.environment in _EXPERIMENTAL_MCP_AUTO_INJECT_ENVIRONMENTS

    def to_config(self) -> "FlowConfig":
        """Return the complete config the graph builder reads, which a full config already is."""
        return self


class PartialFlowConfig(ChatPartialMixin, FlowConfig):
    """A ``chat-partial`` flow config as its author declares it, which may omit ``routers`` and ``flow``."""

    environment: Literal["chat-partial"]
    flow: Optional[FlowConfigMetadata] = None  # type: ignore[assignment]
    routers: Optional[list[dict]] = None  # type: ignore[assignment]

    @override
    def to_config(self) -> FlowConfig:
        """Return the complete ``FlowConfig``, with what the ``chat-partial`` environment lets authors omit."""
        return self._complete_as(FlowConfig)


# The classes that claim an environment in the experimental schema. It leaves
# ``environment`` open, so any other environment validates as ``FlowConfig``.
CONFIG_CLASSES: tuple[type[FlowConfig], ...] = (PartialFlowConfig,)


def config_from_data(data: dict[str, Any], **fields: Any) -> FlowConfig:
    """Validate *data* as the experimental class of its ``environment``."""
    config_cls = select_config_class(
        data.get("environment"), CONFIG_CLASSES, FlowConfig
    )
    return config_cls(**data, **fields)


def load_config(flow_id: str, flow_version: Optional[str] = None) -> FlowConfig:
    """Load an experimental registry file as the class of its ``environment``."""
    return load_registry_config(
        FlowConfig, flow_id, flow_version, parse=config_from_data
    )


def load_config_file(path: Path, flow_id: str, version: str) -> FlowConfig:
    """Load one known experimental registry file as the class of its ``environment``."""
    return load_registry_file(path, flow_id, version, parse=config_from_data)


def load_component_class(
    cls_name: str,
) -> type[BaseComponent] | Callable[..., BaseComponent]:
    """Load a component class by name from the ComponentRegistry.

    This function provides a convenient way to dynamically retrieve registered
    component classes from the global ComponentRegistry instance. It is primarily
    used within the flow system to instantiate components based on their string
    names as specified in flow configuration files.

    The function performs a simple lookup in the ComponentRegistry and returns
    the component class that was previously registered using the @register_component
    decorator or manual registry.register() calls.

    Args:
        cls_name: The name of the component class to load. This should match
            the class name that was used during registration. Component names
            are case-sensitive and must be exact matches.

    Returns:
        The component class registered under the given name. This can be either
        a direct BaseComponent subclass or a callable that returns a BaseComponent
        instance (if decorators were added during registration).

    Raises:
        KeyError: If no component is registered under the given name.

    Example:
        Basic usage in flow configuration:
        >>> component_class = load_component_class("AgentComponent")
        >>> instance = component_class(name="agent", flow_id="flow_1", ...)

    Note:
        This function is typically called internally by the flow system when
        building flows from configuration files. Components must be registered
        before they can be loaded. See `components.register_component` decorator
        for information on how to register components for use with this function.
    """
    registry = ComponentRegistry.instance()

    # pylint: disable-next=unsubscriptable-object
    return registry[cls_name]


def list_configs() -> List[dict[str, str]]:
    return list_flow_configs(FlowConfig, parse=config_from_data)
