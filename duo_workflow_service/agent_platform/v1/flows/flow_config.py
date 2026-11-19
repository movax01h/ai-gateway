import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Self,
    Sequence,
    TypeVar,
    override,
)

import structlog
import yaml
from pydantic import BaseModel, PrivateAttr

from ai_gateway.prompts.config.base import InMemoryPromptConfig
from ai_gateway.response_schemas.config import InlineResponseSchemaConfig
from duo_workflow_service.agent_platform.constants import CHAT_PARTIAL_ENVIRONMENT
from duo_workflow_service.agent_platform.v1.catalog import CatalogItemRef
from duo_workflow_service.agent_platform.v1.components import (
    BaseComponent,
    ComponentRegistry,
)
from duo_workflow_service.agent_platform.v1.components.prompt_reference import (
    prompt_reference_of,
)
from lib.feature_roots import default_features_dir
from lib.version import resolve_version

__all__ = [
    "CONFIG_CLASSES",
    "DEFAULT_FLOW_VERSION",
    "BaseFlowConfig",
    "ChatPartialMixin",
    "FlowConfig",
    "FlowConfigInput",
    "FlowConfigMetadata",
    "PartialFlowConfig",
    "config_from_data",
    "discover_feature_flow_configs",
    "list_configs",
    "list_flow_configs",
    "load_component_class",
    "load_config",
    "load_config_file",
    "load_registry_config",
    "load_registry_file",
    "register_flow_config_root",
    "select_config_class",
]

logger = structlog.stdlib.get_logger(__name__)

_ConfigT = TypeVar("_ConfigT", bound="BaseFlowConfig")

INPUT_JSONSCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema#"
DEFAULT_FLOW_VERSION = "1.0.0"
MCP_AUTO_INJECT_ENVIRONMENTS = frozenset(("chat", "chat-partial"))

# flow_id -> that feature's config/ dir under ai/features/<domain>/<feature>/ (Layout B).
# Only the V1 FlowConfig uses these; the experimental subclass keeps the empty default.
_FEATURE_FLOW_ROOTS: dict[str, Path] = {}


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


def discover_feature_flow_configs(
    features_dir: Optional[Path] = None,
    include: Optional[Callable[[Path], bool]] = None,
) -> None:
    """Register the ``config/`` dir of each feature under ``ai/features/<domain>/<feature>/``.

    A feature without a ``config/`` dir (for example a prompt-only feature) is
    skipped. Idempotent and silent when the tree is absent.

    Args:
        features_dir: Root directory to scan for feature config dirs. Defaults
            to the repo's ``ai/features`` tree when not provided.
        include: Called with each feature dir that has a ``config/`` dir. A
            feature dir it returns ``False`` for is skipped. ``None`` registers
            every feature.

    Raises:
        ValueError: If two features register the same flow id with different
            config directories (see ``register_flow_config_root``).
    """
    root = features_dir or default_features_dir()
    logger.info(
        "Discovering feature flow configs", root=str(root), exists=root.is_dir()
    )
    if not root.is_dir():
        return

    for feature_dir in root.glob("*/*"):
        config_dir = feature_dir / "config"
        if config_dir.is_dir() and (include is None or include(feature_dir)):
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
    # The environments a config class claims within its schema. Empty on a schema's full
    # class, which takes every environment no class in ``CONFIG_CLASSES`` claims.
    SUPPORTED_ENVIRONMENTS: ClassVar[frozenset[str]] = frozenset()

    # Feature ``config/`` subdir for this tier: ``None`` = none, ``""`` = v1.
    FEATURE_CONFIG_SUBDIR: ClassVar[Optional[str]] = None

    @classmethod
    def feature_config_roots(cls) -> dict[str, Path]:
        subdir = cls.FEATURE_CONFIG_SUBDIR
        if subdir is None:
            return {}
        return {
            flow_id: config_dir / subdir
            for flow_id, config_dir in _FEATURE_FLOW_ROOTS.items()
        }

    flow: FlowConfigMetadata
    components: list[dict]
    routers: list[dict]
    environment: str
    version: str
    # The concrete semver this config represents (e.g. "2.1.0"), set when loaded from
    # the registry, from the filename stem. This is the flow's own identity — the
    # version actually run — as opposed to ``version`` (the schema version, e.g. "v1") or
    # the constraint a client requested (e.g. "^2.0.0"). ``None`` for configs built
    # directly (inline flows, tests), which have no registry resolution step.
    resolved_version: Optional[str] = None
    prompts: Optional[list] = None
    response_schemas: Optional[list] = None
    name: Optional[str] = None
    description: Optional[str] = None
    product_group: Optional[str] = None

    # The registry flow id and concrete version this config was loaded from. Set only by
    # ``load_registry_file`` and the loaders that call it, so a config built from
    # caller-supplied data (an inline flow) cannot claim a registry identity. Private
    # attributes are not part of the schema: input cannot set them and ``model_dump``
    # does not emit them.
    _config_id: Optional[str] = PrivateAttr(default=None)
    _config_version: Optional[str] = PrivateAttr(default=None)

    @property
    def config_id(self) -> Optional[str]:
        """The registry flow id this config was loaded from; ``None`` when not loaded from the registry."""
        return self._config_id

    @property
    def config_version(self) -> Optional[str]:
        """The concrete version this config was loaded from; ``None`` when not loaded from the registry."""
        return self._config_version

    @property
    def components_with_repository_prompt(self) -> list[str]:
        """The names of the components that reference a prompt file instead of declaring theirs inline."""
        return [
            component.get("name", "<unnamed>")
            for component in self.components
            if (reference := prompt_reference_of(component)) and reference.is_stored
        ]

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
        """Load a registry file as this class, whatever its ``environment``.

        For tests and tools that know the class a file validates as. The registry loads
        through its schema module's ``load_config``, which picks the class from the file's
        ``environment``.

        Args:
            flow_id: Flow name (e.g. "developer").
            flow_version: Version constraint (e.g. "1.0.0", "^1.0.0"). None uses the default (1.0.0).

        Returns:
            The loaded config, with the registry identity ``load_registry_config`` sets.
        """
        return load_registry_config(
            cls,
            flow_id,
            flow_version,
            parse=lambda data, **fields: cls(**data, **fields),
        )

    @classmethod
    def from_registry_file(cls, path: Path, flow_id: str, version: str) -> Self:
        """Load one registry file as this class, with the identity it has at runtime.

        For tests that already know the file, its version and the class it validates as. The config sweep in CI goes
        through the registry's ``load_config_file``, which picks the class from the file's ``environment``. See
        ``load_registry_file``.

        Args:
            path: The config file.
            flow_id: The registry flow id the file belongs to.
            version: The concrete version the file holds.

        Returns:
            The loaded config, with ``resolved_version`` set to ``version``.
        """
        return load_registry_file(
            path,
            flow_id,
            version,
            parse=lambda data, **fields: cls(**data, **fields),
        )


def resolve_registry_file(
    config_cls: type[BaseFlowConfig], flow_id: str, flow_version: Optional[str]
) -> tuple[Path, str]:
    """Find the registry file of *flow_id* that *flow_version* resolves to.

    Version candidates come from both the feature root (for a flow moved
    under ai/features/) and the legacy ``DIRECTORY_PATH`` root of *config_cls*; on a
    ``(flow, version)`` collision the legacy copy wins, matching
    ``list_flow_configs``.

    Args:
        config_cls: The schema's config class, whose roots to search.
        flow_id: Flow name (e.g. "developer").
        flow_version: Version constraint (e.g. "1.0.0", "^1.0.0"). None uses the default (1.0.0).
            Supports the same constraint syntax as Poetry — see
            https://python-poetry.org/docs/dependency-specification/#version-constraints
            Path traversal is prevented by _safe_resolve.

    Returns:
        The file and the concrete semver the constraint resolved to (e.g. "2.1.0").
    """

    # A registry id is a bare directory name. A path that resolves to the same
    # directory would load the flow under an id it does not match.
    if flow_id in ("", ".", "..") or Path(flow_id).name != flow_id:
        raise ValueError(f"Invalid flow id: '{flow_id}' is not a bare name")

    version_query = flow_version or DEFAULT_FLOW_VERSION

    # Version candidates from the feature root (ai/features/<domain>/<feature>/config/)
    # and the legacy <DIRECTORY_PATH>/<flow_id>/. Legacy wins per version, matching
    # list_flow_configs, so a config fetched into the writable legacy root overrides
    # the bundled feature copy.
    candidates: dict[str, Path] = {}
    feature_config_dir = config_cls.feature_config_roots().get(flow_id)
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
    legacy_base = config_cls.DIRECTORY_PATH.resolve()
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
    return candidates[version], version


def _read_registry_file(path: Path, flow_id: str, version: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"{flow_id}/{version} file not found at {path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e


def load_registry_config(
    config_cls: type[BaseFlowConfig],
    flow_id: str,
    flow_version: Optional[str],
    parse: Callable[..., _ConfigT],
) -> _ConfigT:
    """Load a registry file of *config_cls*'s schema with *parse* and set its registry identity.

    Args:
        config_cls: The schema's config class, whose roots to search.
        flow_id: Flow name (e.g. "developer").
        flow_version: Version constraint; see ``resolve_registry_file``.
        parse: Builds the config from the file's data and ``resolved_version``.

    Returns:
        The config, with ``resolved_version`` and ``config_version`` set to the concrete
        semver the constraint resolved to, and ``config_id`` set to ``flow_id``.
    """
    path, version = resolve_registry_file(config_cls, flow_id, flow_version)
    return load_registry_file(path, flow_id, version, parse)


def load_registry_file(
    path: Path, flow_id: str, version: str, parse: Callable[..., _ConfigT]
) -> _ConfigT:
    """Load one registry file with *parse*, with the identity it has at runtime.

    Every registry load goes through here: ``load_registry_config`` after it resolves the version, and the config
    sweep in CI. An inline flow or a flow from the AI Catalog never does, so it has no identity, which is what rules
    reserved for repository flows read.

    Args:
        path: The config file.
        flow_id: The registry flow id the file belongs to.
        version: The concrete version the file holds.
        parse: Builds the config from the file's data and ``resolved_version``.

    Returns:
        The config, with ``resolved_version`` and ``config_version`` set to *version*, and ``config_id`` set to
        *flow_id*.
    """
    config = parse(
        _read_registry_file(path, flow_id, version) or {}, resolved_version=version
    )
    config._config_id = flow_id
    config._config_version = version
    return config


def select_config_class(
    environment: object,
    environment_classes: Sequence[type[_ConfigT]],
    full_cls: type[_ConfigT],
) -> type[_ConfigT]:
    """Return the class in *environment_classes* that claims *environment*, else *full_cls*.

    A missing or unknown environment selects *full_cls*, whose own validation reports it.
    """
    if isinstance(environment, str):
        for config_cls in environment_classes:
            if environment in config_cls.SUPPORTED_ENVIRONMENTS:
                return config_cls
    return full_cls


def list_flow_configs(
    flow_config_cls: type[BaseFlowConfig], parse: Callable[..., BaseFlowConfig]
) -> List[dict[str, str]]:
    """List all available flow configurations for the given config class.

    Scans the legacy ``DIRECTORY_PATH`` root and every feature root from
    ``feature_config_roots()``, de-duplicating by ``(flow, version)`` with the
    legacy copy winning, matching ``resolve_registry_file``. Errors during loading
    are logged for observability but do not stop processing.

    Args:
        flow_config_cls: FlowConfig class whose roots to scan.
        parse: Builds a config from a file's data, as the schema's registry loading does.

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
        # matching resolve_registry_file, where a malformed legacy file still wins
        # over the feature copy (and then fails to parse).
        seen.add((flow_name, flow_version))
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            config = parse(config_data)
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


class _ChatPartialConfig(Protocol):
    """The parts of a schema's config class that ``ChatPartialMixin`` reads."""

    components: list[dict]
    flow: Optional[FlowConfigMetadata]
    routers: Optional[list[dict]]
    _config_id: Optional[str]
    _config_version: Optional[str]

    def __iter__(self) -> Iterator[tuple[str, Any]]: ...


class ChatPartialMixin:
    """The ``chat-partial`` environment's rules, shared by the v1 and experimental schemas.

    Mix it in before a schema's ``FlowConfig``. It declares no Pydantic fields, so the
    schema keeps its own field types; the partial class declares the fields it relaxes.
    """

    SUPPORTED_ENVIRONMENTS: ClassVar[frozenset[str]] = frozenset(
        {CHAT_PARTIAL_ENVIRONMENT}
    )

    def _complete_as(
        self: _ChatPartialConfig, complete_cls: type[_ConfigT]
    ) -> _ConfigT:
        """Return this config as *complete_cls*, filling in what ``chat-partial`` lets authors omit.

        A missing ``flow.entry_point`` defaults to the single component, and missing or empty ``routers`` to that
        component's route to ``end``, which is also how the engine treats ``routers: []``. A declared entry point and
        non-empty routers are kept. This config is not mutated, and the result keeps its ``config_id`` and
        ``config_version``.

        Raises:
            ValueError: If the entry point is missing and the first component has no ``name`` to default it to.
                Validation doesn't require one, because the legacy chat path runs such a config without it.
        """
        flow = self.flow
        entry_point = flow.entry_point if flow else None
        if entry_point is None:
            first = self.components[0] if self.components else {}
            entry_point = first.get("name")
            if not isinstance(entry_point, str):
                raise ValueError(
                    "A chat-partial config without 'flow.entry_point' needs a first "
                    "component with a 'name'"
                )
            flow = FlowConfigMetadata(
                entry_point=entry_point,
                inputs=flow.inputs if flow else None,
            )

        config = complete_cls.model_validate(
            {
                **dict(self),
                "flow": flow,
                "routers": self.routers or [{"from": entry_point, "to": "end"}],
            }
        )
        # Validation builds the config from public fields only, so carry the private
        # registry identity over explicitly.
        config._config_id = self._config_id
        config._config_version = self._config_version
        return config


# A complete v1 flow config: the graph the builder reads, in any environment.
# ``config_from_data`` loads a ``chat-partial`` config as ``PartialFlowConfig``, whose
# ``to_config()`` completes it into a ``FlowConfig``. A comment, not a docstring, so it
# stays out of the published JSON schema's ``description``.
class FlowConfig(BaseFlowConfig):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"
    FEATURE_CONFIG_SUBDIR: ClassVar[Optional[str]] = ""
    environment: Literal["ambient", "chat", "chat-partial"]
    version: Literal["v1"]
    prompts: Optional[list[InMemoryPromptConfig]] = None
    response_schemas: Optional[list[InlineResponseSchemaConfig]] = None
    # What this flow can pull in from outside. A component claims an entry by
    # repeating it in its `subagents` list; both sides resolve in `bind_catalog_items`.
    include: Optional[list[CatalogItemRef]] = None

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


# The classes that claim an environment in the v1 schema; any other environment
# (``ambient``, ``chat``) validates as ``FlowConfig``.
CONFIG_CLASSES: tuple[type[FlowConfig], ...] = (PartialFlowConfig,)


def config_from_data(data: dict[str, Any], **fields: Any) -> FlowConfig:
    """Validate *data* as the v1 class of its ``environment``.

    Args:
        data: The config as read from YAML, a protobuf Struct, or a request.
        **fields: Extra fields the loader sets, such as ``resolved_version``.

    Returns:
        The validated config; call ``to_config()`` for the graph the builder reads.
    """
    config_cls = select_config_class(
        data.get("environment"), CONFIG_CLASSES, FlowConfig
    )
    return config_cls(**data, **fields)


def load_config(flow_id: str, flow_version: Optional[str] = None) -> FlowConfig:
    """Load a v1 registry file as the class of its ``environment``; see ``resolve_registry_file``."""
    return load_registry_config(
        FlowConfig, flow_id, flow_version, parse=config_from_data
    )


def load_config_file(path: Path, flow_id: str, version: str) -> FlowConfig:
    """Load one known v1 registry file as the class of its ``environment``; see ``load_registry_file``."""
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
    return list_flow_configs(FlowConfig, parse=config_from_data)
