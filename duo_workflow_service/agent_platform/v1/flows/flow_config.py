import json
from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, Self

import structlog
import yaml
from pydantic import BaseModel

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from ai_gateway.response_schemas.config import InlineResponseSchemaConfig
from duo_workflow_service.agent_platform.v1.catalog import CatalogItemRef
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
    "discover_feature_flow_configs",
    "list_configs",
    "list_flow_configs",
    "load_component_class",
    "register_flow_config_root",
]

logger = structlog.stdlib.get_logger(__name__)

INPUT_JSONSCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema#"
DEFAULT_FLOW_VERSION = "1.0.0"
MCP_AUTO_INJECT_ENVIRONMENTS = frozenset(("chat", "chat-partial"))

# flow_id -> that feature's config/ dir under ai/features/<domain>/<feature>/ (Layout B).
# Only the V1 FlowConfig uses these; the experimental subclass keeps the empty default.
_FEATURE_FLOW_ROOTS: dict[str, Path] = {}


def _default_features_dir() -> Path:
    """Return the ``ai/features`` dir (repo root in dev, the WORKDIR in the image).

    Walks up from this file to the nearest ancestor that contains
    ``pyproject.toml``, so a later move of this module cannot silently point
    discovery at the wrong directory. Without a marker (a non-editable
    install, or a faked filesystem in tests) it falls back to the fixed-depth
    derivation: absence of moved features must never fail the caller.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "ai" / "features"
    return here.parents[4] / "ai" / "features"


def register_flow_config_root(flow_id: str, config_dir: Path) -> None:
    """Register a feature's config dir; re-registering the same dir is a no-op.

    Args:
        flow_id: The flat flow id (the feature directory name).
        config_dir: The feature's ``config/`` dir; resolved before storing so two
            spellings of the same directory do not raise a spurious conflict.

    Raises:
        ValueError: If ``flow_id`` is already registered with a different directory.
    """
    config_dir = Path(config_dir).resolve()
    existing = _FEATURE_FLOW_ROOTS.get(flow_id)
    if existing is not None and existing != config_dir:
        raise ValueError(
            f"Duplicate flow config root for {flow_id!r}: {existing} and {config_dir}. "
            "Feature directory names are the flat flow id and must be unique."
        )
    _FEATURE_FLOW_ROOTS[flow_id] = config_dir


def discover_feature_flow_configs(features_dir: Optional[Path] = None) -> None:
    """Register the ``config/`` dir of each feature under ``ai/features/<domain>/<feature>/``.

    A feature without a ``config/`` dir (for example a prompt-only feature) is
    skipped. Idempotent and silent when the tree is absent.

    Args:
        features_dir: Root directory to scan for feature config dirs. Defaults
            to the repo's ``ai/features`` tree when not provided.

    Raises:
        ValueError: If two features register the same flow id with different
            config directories (see ``register_flow_config_root``).
    """
    root = features_dir or _default_features_dir()
    logger.info(
        "Discovering feature flow configs", root=str(root), exists=root.is_dir()
    )
    if not root.is_dir():
        return

    for feature_dir in root.glob("*/*"):
        config_dir = feature_dir / "config"
        if config_dir.is_dir():
            register_flow_config_root(feature_dir.name, config_dir)


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

    @classmethod
    def feature_config_roots(cls) -> dict[str, Path]:
        """Map a flow id to its moved ``config/`` dir.

        Empty unless overridden.

        Returns:
            A dict mapping each registered flow id to its ``config/`` directory.
        """
        return {}

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
                "additionalProperties": True,
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

    def should_auto_inject_mcp_tools(self) -> bool:
        """Return whether MCP tools should be automatically injected into this flow's toolset.

        MCP auto-injection is enabled for ``chat`` and ``chat-partial`` environment flows
        that power interactive chat assistants (e.g. Duo CLI, Interactive Developer,
        Software Development, Agentic Chat, Support Assistant, Analytics Agent).
        ``ambient`` flows receive only the tools explicitly declared in
        their YAML ``toolset:`` — the toolset is fully deterministic.

        Returns:
            ``True`` when the flow's ``environment`` is ``"chat"`` or ``"chat-partial"``,
            ``False`` otherwise.
        """
        return self.environment in MCP_AUTO_INJECT_ENVIRONMENTS

    @classmethod
    def from_yaml_config(cls, flow_id: str, flow_version: Optional[str] = None) -> Self:
        """Load a flow config from its YAML file.

        Version candidates come from both the feature root (for a flow moved
        under ai/features/) and the legacy ``DIRECTORY_PATH`` root; on a
        ``(flow, version)`` collision the legacy copy wins, matching
        ``list_flow_configs``.

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

        # Version candidates from the feature root (ai/features/<domain>/<feature>/config/)
        # and the legacy <DIRECTORY_PATH>/<flow_id>/. Legacy wins per version, matching
        # list_flow_configs, so a config fetched into the writable legacy root overrides
        # the bundled feature copy.
        candidates: dict[str, Path] = {}
        feature_config_dir = cls.feature_config_roots().get(flow_id)
        if feature_config_dir is not None:
            feature_base = feature_config_dir.resolve()
            for f in feature_base.glob("*.yml"):
                try:
                    candidates[f.stem] = _safe_resolve(f, feature_base)
                except ValueError as e:
                    logger.warning(
                        "Security validation failed for config file",
                        config_file=str(f),
                        error=str(e),
                    )
        legacy_base = cls.DIRECTORY_PATH.resolve()
        try:
            legacy_dir = _safe_resolve(legacy_base / flow_id, legacy_base)
        except ValueError as e:
            # An unsafe legacy dir must not discard the feature candidates:
            # list_flow_configs skips it per file and still lists the feature
            # copy, and loading must agree with the listing.
            if not candidates:
                raise
            logger.warning(
                "Security validation failed for legacy flow dir",
                flow_id=flow_id,
                error=str(e),
            )
        else:
            for f in legacy_dir.glob("*.yml"):
                try:
                    candidates[f.stem] = _safe_resolve(f, legacy_base)
                except ValueError as e:
                    logger.warning(
                        "Security validation failed for config file",
                        config_file=str(f),
                        error=str(e),
                    )

        version = resolve_version(list(candidates), version_query)
        # resolve_version only returns a member of its input, so the lookup
        # cannot miss; a file deleted in between still hits the handler below.
        yaml_path = candidates[version]
        try:
            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_content = yaml.safe_load(file)
            return cls(**yaml_content, resolved_version=version)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{flow_id}/{version} file not found at {yaml_path}"
            )
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e


def list_flow_configs(flow_config_cls: type[BaseFlowConfig]) -> List[dict[str, str]]:
    """List all available flow configurations for the given config class.

    Scans the legacy ``DIRECTORY_PATH`` root and every feature root from
    ``feature_config_roots()``, de-duplicating by ``(flow, version)`` with the
    legacy copy winning, matching ``from_yaml_config``. Errors during loading
    are logged for observability but do not stop processing.

    Args:
        flow_config_cls: FlowConfig class whose roots to scan.

    Returns:
        List of dicts containing flow metadata and JSON-serialized configuration.
    """
    configs = []
    seen: set[tuple[str, str]] = set()

    # (flow_identifier, config_file, base_path) across the legacy root and each moved
    # feature's config/ dir. Legacy first, so it wins on a (flow, version) collision.
    entries: list[tuple[str, Path, Path]] = []
    legacy_base = flow_config_cls.DIRECTORY_PATH.resolve()
    for config_file in flow_config_cls.DIRECTORY_PATH.glob("*/*.yml"):
        entries.append((config_file.parent.name, config_file, legacy_base))
    for flow_name, config_dir in flow_config_cls.feature_config_roots().items():
        feature_base = config_dir.resolve()
        for config_file in config_dir.glob("*.yml"):
            entries.append((flow_name, config_file, feature_base))

    for flow_name, config_file, base_path in entries:
        flow_version = config_file.stem  # e.g. "1.0.0" from "1.0.0.yml"
        if (flow_name, flow_version) in seen:
            continue
        try:
            yaml_path = _safe_resolve(config_file, base_path)
        except ValueError as e:
            logger.warning(
                "Security validation failed for config file",
                config_file=str(config_file),
                error=str(e),
            )
            continue

        # A path-safe entry shadows later duplicates even when it fails to load,
        # matching from_yaml_config, where a malformed legacy file still wins
        # over the feature copy (and then fails to parse).
        seen.add((flow_name, flow_version))
        try:
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
    # What this flow can pull in from outside. A component claims an entry by
    # repeating it in its `subagents` list; both sides resolve in `bind_catalog_items`.
    include: Optional[list[CatalogItemRef]] = None

    @classmethod
    def feature_config_roots(cls) -> dict[str, Path]:
        return dict(_FEATURE_FLOW_ROOTS)


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
