from pathlib import Path
from typing import Callable, ClassVar, List, Optional

from duo_workflow_service.agent_platform.experimental.components import (
    BaseComponent,
    ComponentRegistry,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    BaseFlowConfig,
    FlowConfigInput,
    FlowConfigMetadata,
    list_flow_configs,
)

__all__ = [
    "FlowConfig",
    "FlowConfigInput",
    "FlowConfigMetadata",
    "PartialFlowConfig",
    "load_component_class",
    "list_configs",
]


class FlowConfig(BaseFlowConfig):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"


class PartialFlowConfig(FlowConfig):
    flow: Optional[FlowConfigMetadata] = None  # type: ignore[assignment]
    routers: Optional[list[dict]] = None  # type: ignore[assignment]


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
    return list_flow_configs(FlowConfig)
