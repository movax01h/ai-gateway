import json
from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, Self

import structlog
import yaml
from pydantic import BaseModel

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from ai_gateway.response_schemas.config import InlineResponseSchemaConfig
from duo_workflow_service.agent_platform.v1.components import (
    BaseComponent,
    ComponentRegistry,
)
from lib.version import resolve_version

__all__ = [
    "DEFAULT_FLOW_VERSION",
    "BaseFlowConfig",
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


def _safe_resolve(path: Path, base_path: Path) -> Path:
    """Resolve *path*, allowing symlinks only when they stay within *base_path*.

    Symlink check must happen before .resolve() because resolve() follows
    symlinks, after which is_symlink() always returns False.

    Symlinks that resolve to a path inside *base_path* are permitted so that
    flow config files can reference other configs within the same directory
    tree (e.g. ``developer_unstable/1.0.0.yml`` → ``../developer/2.0.0.yml``).
    Symlinks that escape *base_path* are rejected to prevent path-traversal
    attacks.

    Raises:
        ValueError: If path resolves outside base_path, or if symlink resolution
            fails (e.g. circular symlinks raise OSError/RuntimeError).
    """
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Symlink resolution failed for '{path.name}': {e}") from e
    if not resolved.is_relative_to(base_path):
        raise ValueError(
            f"Path traversal detected: '{path.name}' resolves outside config directory"
        )
    return resolved


class FlowConfigInputSchema(BaseModel):
    type: str
    format: Optional[str] = None
    description: Optional[str] = None
    optional: Optional[bool] = None


class FlowConfigInput(BaseModel):
    category: str
    input_schema: dict[str, FlowConfigInputSchema]
    version_constraint: Optional[str] = None


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
    # The concrete semver this config represents (e.g. "2.1.0"), set when loaded via
    # ``from_yaml_config`` from the filename stem. This is the flow's own identity — the
    # version actually run — as opposed to ``version`` (the schema version, e.g. "v1") or
    # the constraint a client requested (e.g. "^2.0.0"). ``None`` for configs built
    # directly (inline flows, tests), which have no registry resolution step.
    resolved_version: Optional[str] = None
    prompts: Optional[list] = None
    response_schemas: Optional[list] = None
    name: Optional[str] = None
    description: Optional[str] = None
    product_group: Optional[str] = None

    def input_json_schemas_by_category(self):
        json_schemas_by_category: dict[str, dict] = {}
        if not self.flow.inputs:
            return json_schemas_by_category

        for item in self.flow.inputs:
            schema = {
                key: value.model_dump(exclude_none=True, exclude={"optional"})
                for key, value in item.input_schema.items()
            }
            required_keys = [
                key for key, value in item.input_schema.items() if not value.optional
            ]

            jsonschema = {
                "$schema": INPUT_JSONSCHEMA_VERSION,
                "additionalProperties": False,
                "type": "object",
                "properties": schema,
                "required": required_keys,
            }

            json_schemas_by_category[item.category] = jsonschema

        return json_schemas_by_category

    def version_constraints_by_category(self) -> dict[str, Optional[str]]:
        """Return a mapping of input category to its declared version constraint.

        Returns:
            A dict mapping each input category to its ``version_constraint`` string
            (e.g. ``"^1.0.0"``), or ``None`` when no constraint was declared.
        """
        if not self.flow.inputs:
            return {}
        return {item.category: item.version_constraint for item in self.flow.inputs}

    @classmethod
    def from_yaml_config(cls, flow_id: str, flow_version: Optional[str] = None) -> Self:
        """Load a flow config from its YAML file.

        Args:
            flow_id: Flow name (e.g. "developer").
            flow_version: Version constraint (e.g. "1.0.0", "^1.0.0"). None uses the default (1.0.0).
                Supports the same constraint syntax as Poetry — see
                https://python-poetry.org/docs/dependency-specification/#version-constraints
                Path traversal is prevented by _safe_resolve.

        Returns:
            The loaded config, with ``resolved_version`` set to the concrete semver the
            constraint resolved to (e.g. "2.1.0").
        """
        version_query = flow_version or DEFAULT_FLOW_VERSION
        base_path = cls.DIRECTORY_PATH.resolve()
        flow_dir = _safe_resolve(base_path / flow_id, base_path)
        available = []
        for f in flow_dir.glob("*.yml"):
            try:
                _safe_resolve(f, base_path)
                available.append(f.stem)
            except ValueError:
                pass
        version = resolve_version(available, version_query)
        try:
            yaml_path = _safe_resolve(flow_dir / f"{version}.yml", base_path)
            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_content = yaml.safe_load(file)
            return cls(**yaml_content, resolved_version=version)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{flow_id}/{version} file not found in {cls.DIRECTORY_PATH}"
            )
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
    for config_file in flow_config_cls.DIRECTORY_PATH.glob("*/*.yml"):
        flow_name = config_file.parent.name
        flow_version = config_file.stem  # e.g. "1.0.0" from "1.0.0.yml"
        try:
            base_path = flow_config_cls.DIRECTORY_PATH.resolve()
            yaml_path = _safe_resolve(config_file, base_path)

            with open(yaml_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            config = flow_config_cls(**config_data)
            configs.append(
                {
                    "flow_identifier": flow_name,
                    "flow_version": flow_version,
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
    response_schemas: Optional[list[InlineResponseSchemaConfig]] = None


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
