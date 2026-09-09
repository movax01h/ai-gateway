"""Feature-owned serving-surface declarations and their collector (baseline).

A feature declares how it is served — the transport (gRPC / REST) and the
deployable (AI Gateway / Duo Workflow Service) — in a ``serving.py`` module
under its feature directory. The collector aggregates these declarations so the
per-app composition can later query which features each deployable serves.

This is the baseline form (declaration + collector). Consuming the collector
from a per-app container is completed in the platform epic (#47).
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, get_args

import structlog

logger = structlog.stdlib.get_logger(__name__)

Transport = Literal["grpc", "rest"]
Deployable = Literal["aigw", "dws"]

__all__ = [
    "Deployable",
    "ServingSurface",
    "Transport",
    "collect_serving_surfaces",
]


@dataclass(frozen=True)
class ServingSurface:
    """One transport/deployable pair a feature is reachable through."""

    transport: Transport
    deployable: Deployable


def _validate_declaration(declared: object, serving_file: Path) -> bool:
    """Report whether a ``SERVING_SURFACE`` declaration is well-formed.

    ``Literal`` types are not enforced at runtime, so a plain tuple or a typo'd
    transport/deployable string would otherwise pass through and fail far away
    in a consumer.
    """
    if not isinstance(declared, (list, tuple)):
        logger.warning(
            "Skipping serving file: SERVING_SURFACE must be a list or tuple",
            path=str(serving_file),
        )
        return False
    for item in declared:
        if not isinstance(item, ServingSurface):
            logger.warning(
                "Skipping serving file: SERVING_SURFACE item is not a ServingSurface",
                path=str(serving_file),
                item=repr(item),
            )
            return False
        if item.transport not in get_args(Transport) or item.deployable not in get_args(
            Deployable
        ):
            logger.warning(
                "Skipping serving file: unknown transport or deployable",
                path=str(serving_file),
                transport=item.transport,
                deployable=item.deployable,
            )
            return False
    return True


def _features_dir() -> Path:
    """Return the ``ai/features`` dir (repo root in dev, the WORKDIR in the image).

    Walks up from this file to the nearest ancestor that contains
    ``pyproject.toml``, so a later move of this module cannot silently point
    discovery at the wrong directory.

    Without a marker (a non-editable install, or a faked filesystem in tests)
    it falls back to the fixed-depth derivation: absence of moved features
    must never fail the caller.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "ai" / "features"
    return here.parents[2] / "ai" / "features"


def collect_serving_surfaces(
    features_dir: Optional[Path] = None,
) -> dict[str, list[ServingSurface]]:
    """Return ``{feature_id: [ServingSurface, ...]}`` declared under ai/features/.

    A feature declares a ``SERVING_SURFACE`` list in its ``serving.py`` module.
    A feature without one — or with a malformed one — is skipped and logged
    with the offending path. Silent when the tree is absent.

    Args:
        features_dir: Root to scan for feature ``serving.py`` files. Defaults to
            the repository's ``ai/features`` directory.

    Returns:
        Mapping of feature id to its declared ``ServingSurface`` list.

    Raises:
        ValueError: If two features declare the same feature id.
    """
    root = features_dir or _features_dir()
    surfaces: dict[str, list[ServingSurface]] = {}
    if not root.is_dir():
        return surfaces

    root_resolved = root.resolve()
    for serving_file in root.glob("*/*/serving.py"):
        # exec_module runs the file, so refuse a symlink that escapes the tree.
        try:
            resolved = serving_file.resolve()
        except (OSError, RuntimeError):
            logger.warning("Skipping unresolvable serving file", path=str(serving_file))
            continue
        if not resolved.is_relative_to(root_resolved):
            logger.warning(
                "Skipping serving file outside the features root",
                path=str(serving_file),
            )
            continue
        feature = serving_file.parent.name
        domain = serving_file.parent.parent.name
        spec = importlib.util.spec_from_file_location(
            f"_feature_serving_{domain}_{feature}", resolved
        )
        if spec is None or spec.loader is None:
            logger.warning(
                "Skipping serving file: could not build an import spec",
                path=str(serving_file),
            )
            continue
        module = importlib.util.module_from_spec(spec)
        # One broken declaration must not take down collection for every
        # other feature; skip it and let its owner see the warning.
        try:
            spec.loader.exec_module(module)
        except (Exception, SystemExit):  # pylint: disable=broad-exception-caught
            logger.warning(
                "Skipping serving file that failed to import",
                path=str(serving_file),
                exc_info=True,
            )
            continue
        declared = getattr(module, "SERVING_SURFACE", None)
        if not _validate_declaration(declared, serving_file):
            continue
        if not declared:
            # declared but empty: same as absent
            continue
        if feature in surfaces:
            raise ValueError(
                f"Duplicate serving-surface declaration for feature {feature!r}. "
                "Feature directory names must be unique across ai/features/."
            )
        surfaces[feature] = list(declared)
    return surfaces
