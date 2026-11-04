"""The prompt directories and prompt version files that the prompt registry loads, for tests that check every prompt."""

from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "feature_prompt_dirs",
    "prompt_dir",
    "prompt_ids_by_dir",
    "version_files",
]

REPO_ROOT = Path(__file__).parents[2]
_LEGACY_DIR = REPO_ROOT / "ai_gateway" / "prompts" / "definitions"
_FEATURES_DIR = REPO_ROOT / "ai" / "features"
_SHARED_DIR = REPO_ROOT / "ai" / "shared"


def version_files(prompts_dir: Path) -> list[Path]:
    """Return the ``<model family or base>/<version>.yml`` files in ``prompts_dir``."""
    return sorted(prompts_dir.glob("*/*.yml"))


def feature_prompt_dirs() -> dict[str, Path]:
    """Return the `prompts/` dir of each prompt ID that `discover_feature_prompts` registers and that loads a prompt.

    A flow-only feature keeps only Jinja fragments in its `prompts/` dir. Its ID loads no prompt.
    """
    dirs: dict[str, Path] = {}
    for feature_dir in _FEATURES_DIR.glob("*/*"):
        if version_files(prompts_dir := feature_dir / "prompts"):
            dirs[feature_dir.name] = prompts_dir
            dirs[f"{feature_dir.parent.name}/{feature_dir.name}"] = prompts_dir
    for shared_dir in _SHARED_DIR.glob("*"):
        if version_files(prompts_dir := shared_dir / "prompts"):
            dirs[shared_dir.name] = prompts_dir
    return dirs


def prompt_dir(prompt_id: str) -> Path:
    """Return the directory that the registry resolves ``prompt_id`` to."""
    return feature_prompt_dirs().get(prompt_id, _LEGACY_DIR / prompt_id)


def prompt_ids_by_dir() -> dict[Path, set[str]]:
    """Return every prompt directory, with the prompt IDs that resolve to it."""
    prompt_ids: dict[Path, set[str]] = {}
    for version_file in _LEGACY_DIR.rglob("*.yml"):
        # <prompt_id>/<model family or base>/<version>.yml
        parts = version_file.relative_to(_LEGACY_DIR).parts[:-2]
        if parts:
            prompt_ids.setdefault(_LEGACY_DIR.joinpath(*parts), set()).add(
                "/".join(parts)
            )
    for prompt_id, prompts_dir in feature_prompt_dirs().items():
        prompt_ids.setdefault(prompts_dir, set()).add(prompt_id)
    return prompt_ids
