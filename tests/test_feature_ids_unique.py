# pylint: disable=file-naming-for-tests
"""Structural guard: ids under ai/features/ and ai/shared/ must be unique.

A feature's directory name is its id in the prompt and flow-config registries, and
an ai/shared/ directory name is a prompt id. Two directories sharing an id would
fail at boot, so this catches the collision statically in CI.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURES_DIR = _REPO_ROOT / "ai" / "features"
_SHARED_DIR = _REPO_ROOT / "ai" / "shared"
_LEGACY_PROMPT_DEFS = _REPO_ROOT / "ai_gateway" / "prompts" / "definitions"


def _dirs(pattern: str, root: Path) -> list[Path]:
    return [p for p in root.glob(pattern) if p.is_dir() and p.name != "__pycache__"]


def test_feature_ids_are_unique_across_domains():
    seen: dict[str, Path] = {}
    for directory in _dirs("*/*", _FEATURES_DIR) + _dirs("*", _SHARED_DIR):
        assert directory.name not in seen, (
            f"Duplicate id {directory.name!r}: {seen[directory.name]} and "
            f"{directory}. Feature and shared directory names are registry ids and "
            f"must be unique across ai/features/ and ai/shared/."
        )
        seen[directory.name] = directory


def test_registered_prompt_ids_do_not_shadow_legacy_prompts():
    moved = [
        (d, [d.name, f"{d.parent.name}/{d.name}"]) for d in _dirs("*/*", _FEATURES_DIR)
    ] + [(d, [d.name]) for d in _dirs("*", _SHARED_DIR)]
    for directory, prompt_ids in moved:
        if not (directory / "prompts").is_dir():
            continue
        for prompt_id in prompt_ids:
            legacy = _LEGACY_PROMPT_DEFS / prompt_id
            assert not legacy.exists(), (
                f"Prompt id {prompt_id!r} is registered from {directory} and also "
                f"exists in the legacy root at {legacy}. The registered copy hides "
                f"the legacy one. Remove the legacy copy after a move, or rename one "
                f"of the two prompts."
            )
