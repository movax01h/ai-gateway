"""Location of the project root and the ``ai/features`` tree, shared by the gateway and the workflow service."""

from pathlib import Path
from typing import NamedTuple

__all__ = [
    "FOUNDATIONAL_AGENTS_DOMAIN",
    "FeatureSubdir",
    "default_features_dir",
    "default_project_root",
    "feature_subdirs",
]

# Domain under ai/features/ that fetch-foundational-agents writes AI Catalog flows into.
FOUNDATIONAL_AGENTS_DOMAIN = "foundational_agents"

# package depth of this module -> parents index of the project root
_ROOT_DEPTH = __name__.count(".")


def default_project_root() -> Path:
    """Return the project root (repo root in dev, the WORKDIR in the image).

    Walks up from this file to the nearest ancestor that contains
    ``pyproject.toml``, so a later move of this module cannot silently point
    callers at the wrong directory. Without a marker (a non-editable
    install, or a faked filesystem in tests) it falls back to the fixed-depth
    derivation: absence of moved features must never fail construction.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[_ROOT_DEPTH]


def default_features_dir() -> Path:
    """Return the ``ai/features`` dir under :func:`default_project_root`."""
    return default_project_root() / "ai" / "features"


class FeatureSubdir(NamedTuple):
    """One ``<subdir>`` dir of a feature or of a shared definition."""

    path: Path
    name: str
    domain: str | None  # None under ai/shared/

    @property
    def shared(self) -> bool:
        return self.domain is None


def feature_subdirs(
    subdir: str, features_dir: Path | None = None
) -> list[FeatureSubdir]:
    """Return each ``<subdir>`` dir under ``ai/features/*/*/`` and ``ai/shared/*/``.

    Feature dirs come first, then shared dirs, each sorted by path.

    Args:
        subdir: The dir name to find, for example ``prompts`` or ``schemas``.
        features_dir: The ``ai/features`` dir. ``ai/shared`` is its sibling.
            Defaults to :func:`default_features_dir`.
    """
    root = features_dir or default_features_dir()
    found = [
        FeatureSubdir(path, path.parent.name, path.parent.parent.name)
        for path in sorted(root.glob(f"*/*/{subdir}"))
        if path.is_dir()
    ]
    found += [
        FeatureSubdir(path, path.parent.name, None)
        for path in sorted((root.parent / "shared").glob(f"*/{subdir}"))
        if path.is_dir()
    ]
    return found
