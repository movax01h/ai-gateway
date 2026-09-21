"""Location of the ``ai/features`` tree, shared by the gateway and the workflow service."""

from pathlib import Path

__all__ = ["default_features_dir"]

# package depth of this module -> parents index of the project root
_ROOT_DEPTH = __name__.count(".")


def default_features_dir() -> Path:
    """Return the ``ai/features`` dir (repo root in dev, the WORKDIR in the image).

    Walks up from this file to the nearest ancestor that contains
    ``pyproject.toml``, so a later move of this module cannot silently point
    discovery at the wrong directory. Without a marker (a non-editable
    install, or a faked filesystem in tests) it falls back to the fixed-depth
    derivation: absence of moved features must never fail construction.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "ai" / "features"
    return here.parents[_ROOT_DEPTH] / "ai" / "features"
