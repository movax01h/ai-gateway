"""Registry of moved feature prompt roots (module-boundary Layout B).

A feature that has moved under ``ai/features/<domain>/<feature>/`` owns its prompts
in ``ai/features/<domain>/<feature>/prompts/``. Both discovery paths consult this
registry by the flat prompt id:

- version-file discovery in ``LocalPromptRegistry._resolve_id``
- Jinja ``{% include %}`` resolution via ``FeatureRootLoader`` in ``base.py``

Registration strips the feature-id prefix, so a self-namespaced include such as
``glab_ask_git_command/system/1.0.0.jinja`` keeps working unchanged.
"""

from pathlib import Path

__all__ = [
    "default_features_dir",
    "discover_feature_prompts",
    "feature_prompt_root",
    "register_prompt_root",
]

# flat prompt id -> that feature's `prompts/` directory
_FEATURE_PROMPT_ROOTS: dict[str, Path] = {}


def register_prompt_root(feature_id: str, prompts_dir: Path) -> None:
    """Register a moved feature's ``prompts/`` dir under its flat prompt id.

    Args:
        feature_id: The flat prompt id (the feature directory name).
        prompts_dir: The feature's ``prompts/`` directory.

    Raises:
        ValueError: If ``feature_id`` is already registered to a different path.
            The prompt-id namespace is flat, so two features sharing an id would
            resolve nondeterministically; raising surfaces the collision instead.
    """
    prompts_dir = Path(prompts_dir)
    existing = _FEATURE_PROMPT_ROOTS.get(feature_id)
    if existing is not None and existing != prompts_dir:
        raise ValueError(
            f"Duplicate prompt feature id {feature_id!r}: already registered at "
            f"{existing}, cannot also register {prompts_dir}"
        )
    _FEATURE_PROMPT_ROOTS[feature_id] = prompts_dir


def feature_prompt_root(feature_id: str) -> Path | None:
    """Return the registered ``prompts/`` dir for a flat prompt id, or ``None``."""
    return _FEATURE_PROMPT_ROOTS.get(feature_id)


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
    return here.parents[2] / "ai" / "features"


def discover_feature_prompts(features_dir: Path | None = None) -> None:
    """Register the ``prompts/`` dir of each feature under ``ai/features/<domain>/<feature>/``.

    A feature without a ``prompts/`` dir (for example a flow-only feature) is
    skipped. Idempotent and silent when the tree is absent (the default before
    any feature moves).

    Args:
        features_dir: Root directory to scan for features. Defaults to
            ``default_features_dir()`` when not provided.
    """
    root = features_dir or default_features_dir()
    if not root.is_dir():
        return

    for feature_dir in root.glob("*/*"):
        prompts_dir = feature_dir / "prompts"
        if prompts_dir.is_dir():
            register_prompt_root(feature_dir.name, prompts_dir)
