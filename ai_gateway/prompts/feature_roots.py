"""Registry of prompt roots under ``ai/features/`` and ``ai/shared/``.

A feature owns its prompts in ``ai/features/<domain>/<feature>/prompts/``. A
definition that several features use lives in ``ai/shared/<name>/prompts/``.
Both discovery paths consult this registry by prompt id:

- version-file discovery in ``LocalPromptRegistry._resolve_id``
- Jinja ``{% include %}`` resolution via ``FeatureRootLoader`` in ``base.py``

Registration strips the prompt-id prefix, so a self-namespaced include such as
``glab_ask_git_command/system/1.0.0.jinja`` keeps working unchanged.
"""

from pathlib import Path

from lib.feature_roots import default_features_dir

__all__ = [
    "discover_feature_prompts",
    "feature_prompt_root",
    "register_prompt_root",
]

# prompt id -> the `prompts/` directory that owns it
_FEATURE_PROMPT_ROOTS: dict[str, Path] = {}


def register_prompt_root(feature_id: str, prompts_dir: Path) -> None:
    """Register a ``prompts/`` dir under a prompt id.

    Args:
        feature_id: The prompt id, for example ``glab_ask_git_command`` or
            ``chat/react``.
        prompts_dir: The ``prompts/`` directory that owns the id.

    Raises:
        ValueError: If ``feature_id`` is already registered to a different path.
            Two directories sharing an id would resolve nondeterministically;
            raising surfaces the collision instead.
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
    """Return the registered ``prompts/`` dir for a prompt id, or ``None``."""
    return _FEATURE_PROMPT_ROOTS.get(feature_id)


def discover_feature_prompts(features_dir: Path | None = None) -> None:
    """Register the ``prompts/`` dirs under ``ai/features/`` and ``ai/shared/``.

    ``ai/features/<domain>/<feature>/prompts/`` registers as ``<feature>`` and as
    ``<domain>/<feature>``, so a nested id such as ``chat/react`` keeps its name
    after it moves to ``ai/features/chat/react/``. ``ai/shared/<name>/prompts/``
    registers as ``<name>``. A directory without ``prompts/`` (for example a
    flow-only feature) is skipped. Idempotent and silent when a tree is absent.

    Args:
        features_dir: The ``ai/features`` directory to scan. ``ai/shared`` is its
            sibling. Defaults to ``default_features_dir()`` when not provided.

    Raises:
        ValueError: If two directories register the same prompt id.
    """
    root = features_dir or default_features_dir()
    shared_root = root.parent / "shared"

    # Each tree is guarded on its own: ai/shared/ can exist without ai/features/.
    if root.is_dir():
        for feature_dir in sorted(root.glob("*/*")):
            prompts_dir = feature_dir / "prompts"
            if prompts_dir.is_dir():
                register_prompt_root(feature_dir.name, prompts_dir)
                register_prompt_root(
                    f"{feature_dir.parent.name}/{feature_dir.name}", prompts_dir
                )

    if shared_root.is_dir():
        for shared_dir in sorted(shared_root.glob("*")):
            prompts_dir = shared_dir / "prompts"
            if prompts_dir.is_dir():
                register_prompt_root(shared_dir.name, prompts_dir)
