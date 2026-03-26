import json
from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, Self

import structlog
import yaml
from pydantic import BaseModel

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from duo_workflow_service.agent_platform.v1.components import (
    BaseComponent,
    ComponentRegistry,
)

__all__ = [
    "BaseFlowConfig",
    "DEFAULT_FLOW_VERSION",
    "FlowConfig",
    "FlowConfigInput",
    "FlowConfigMetadata",
    "PartialFlowConfig",
    "list_configs",
    "list_flow_configs",
    "load_component_class",
]

logger = structlog.stdlib.get_logger(__name__)

INPUT_JSONSCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema#"
DEFAULT_FLOW_VERSION = "1.0.0"
_DEFAULT_FLOW_VERSION = DEFAULT_FLOW_VERSION  # backward-compat alias


def _safe_resolve(path: Path, base_path: Path) -> Path:
    """Resolve *path*, rejecting symlinks and traversal outside *base_path*.

    Symlink check must happen before .resolve() because resolve() follows
    symlinks, after which is_symlink() always returns False.

    Raises:
        ValueError: If path is a symlink or resolves outside base_path.
    """
    if path.is_symlink():
        raise ValueError(f"Symlinks are not allowed for security reasons: {path.name}")
    resolved = path.resolve()
    if not resolved.is_relative_to(base_path):
        raise ValueError(
            f"Path traversal detected: '{path.name}' resolves outside config directory"
        )
    return resolved


class FlowConfigInputSchema(BaseModel):
    type: str
    format: Optional[str] = None
    description: Optional[str] = None


class FlowConfigInput(BaseModel):
    category: str
    input_schema: dict[str, FlowConfigInputSchema]


class FlowConfigMetadata(BaseModel):
    entry_point: Optional[str] = None
    inputs: Optional[list[FlowConfigInput]] = None


class BaseFlowConfig(BaseModel):
    DIRECTORY_PATH: ClassVar[Path]

    flow: FlowConfigMetadata
    components: list[dict]
    routers: list[dict]
    environment: str
    version: str
    prompts: Optional[list] = None
    name: Optional[str] = None
    description: Optional[str] = None
    product_group: Optional[str] = None

    def input_json_schemas_by_category(self):
        json_schemas_by_category: dict[str, dict] = {}
        if not self.flow.inputs:
            return json_schemas_by_category

        for item in self.flow.inputs:
            schema = {
                key: value.model_dump(exclude_none=True)
                for key, value in item.input_schema.items()
            }

            jsonschema = {
                "$schema": INPUT_JSONSCHEMA_VERSION,
                "additionalProperties": False,
                "type": "object",
                "properties": schema,
                "required": list(schema.keys()),
            }

            json_schemas_by_category[item.category] = jsonschema

        return json_schemas_by_category

    @classmethod
    def from_yaml_config(cls, path: str) -> Self:
        try:
            base_path = cls.DIRECTORY_PATH.resolve()
            yaml_path = _safe_resolve(
                base_path / path / f"{_DEFAULT_FLOW_VERSION}.yml", base_path
            )

            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_content = yaml.safe_load(file)

            return cls(**yaml_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} file not found in {cls.DIRECTORY_PATH}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e


def list_flow_configs(flow_config_cls: type[BaseFlowConfig]) -> List[dict[str, str]]:
    """List all available flow configurations for the given config class.

    Errors during loading are logged for observability but do not stop processing.

    Args:
        flow_config_cls: FlowConfig class whose DIRECTORY_PATH to scan.

    Returns:
        List of dicts containing flow metadata and JSON-serialized configuration.
    """
    configs = []
    for config_file in flow_config_cls.DIRECTORY_PATH.glob(
        f"*/{_DEFAULT_FLOW_VERSION}.yml"
    ):
        flow_name = config_file.parent.name
        try:
            base_path = flow_config_cls.DIRECTORY_PATH.resolve()
            yaml_path = _safe_resolve(config_file, base_path)

            with open(yaml_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            config = flow_config_cls(**config_data)
            configs.append(
                {
                    "flow_identifier": flow_name,
                    "version": config.version,
                    "environment": config.environment,
                    "config": json.dumps(config_data, indent=2),
                }
            )
        except yaml.YAMLError as e:
            logger.warning(
                "Failed to parse YAML config file",
                config_file=str(config_file),
                error=str(e),
            )
        except (IOError, OSError) as e:
            logger.warning(
                "Failed to read config file",
                config_file=str(config_file),
                error=str(e),
            )
        except ValueError as e:
            logger.warning(
                "Security validation failed for config file",
                config_file=str(config_file),
                error=str(e),
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error loading config file",
                config_file=str(config_file),
                error=str(e),
                exc_info=True,
            )

    return configs


class FlowConfig(BaseFlowConfig):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"
    environment: Literal["ambient", "chat", "chat-partial"]
    version: Literal["v1"]
    prompts: Optional[list[InMemoryPromptConfig]] = None


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
        instance (if decorators were applied during registration).

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
