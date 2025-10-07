import json
from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, Self

import yaml
from pydantic import BaseModel

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from duo_workflow_service.agent_platform.v1.components import (
    BaseComponent,
    ComponentRegistry,
)

__all__ = ["FlowConfig", "load_component_class", "list_configs"]


_PREFIX_BLOCLIST = (
    "..",
    "/.../",
    r"\…..\\",
    "%00../../../../../",
    "%2e%2e%2f",
    "%252e%252e%252f",
    "%c0%ae%c0%ae%c0%af",
    "%uff0e%uff0e%u2215",
    "%uff0e%uff0e%u2216",
)


INPUT_JSONSCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema#"


class FlowConfigInputSchema(BaseModel):
    type: str
    format: Optional[str] = None
    description: Optional[str] = None


class FlowConfigInput(BaseModel):
    category: str
    input_schema: dict[str, FlowConfigInputSchema]


class FlowConfigMetadata(BaseModel):
    entry_point: str
    inputs: Optional[list[FlowConfigInput]] = None


class FlowConfig(BaseModel):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"
    flow: FlowConfigMetadata
    components: list[dict]
    routers: list[dict]
    environment: Literal["ambient", "chat", "chat-partial"]
    version: Literal["v1"]
    prompts: Optional[list[InMemoryPromptConfig]] = None

    def input_json_schemas_by_category(self):
        json_schemas_by_category: dict[str, dict] = {}
        if not self.flow.inputs:
            return json_schemas_by_category

        for item in self.flow.inputs:
            schema = {
                key: value.model_dump(exclude_none=True)
                for key, value in item.input_schema.items()
            }

            # Create standard jsonschema structure,
            # with all properties being required.
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
            # Validate path before resolving to prevent directory traversal
            if any(prefix in path for prefix in _PREFIX_BLOCLIST) or path.startswith(
                "/"
            ):
                raise ValueError(f"Path traversal detected: {path}")

            base_path = cls.DIRECTORY_PATH.resolve()
            yaml_path = (base_path / f"{path}.yml").resolve()

            if not yaml_path.is_relative_to(base_path):
                raise ValueError(f"Path traversal detected: {path}")

            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_content = yaml.safe_load(file)

            return cls(**yaml_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} file not found in {cls.DIRECTORY_PATH}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e


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
    configs = []
    for config_file in FlowConfig.DIRECTORY_PATH.glob("*.yml"):
        try:
            config = FlowConfig.from_yaml_config(config_file.stem)
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            config_json = json.dumps(config_data, indent=2)
            configs.append(
                {
                    "name": config_file.stem,
                    "version": config.version,
                    "environment": config.environment,
                    "config": config_json,
                }
            )
        except (yaml.YAMLError, IOError):
            continue

    return configs
