# pylint: disable=file-naming-for-tests
"""Structural guard: feature ids under ai/features/ must be unique.

A feature's directory name is its flat id in the prompt and flow-config registries.
Two features sharing an id would resolve nondeterministically at runtime, so this
catches the collision statically in CI instead of at first request.
"""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURES_DIR = _REPO_ROOT / "ai" / "features"
_LEGACY_PROMPT_DEFS = _REPO_ROOT / "ai_gateway" / "prompts" / "definitions"


def _feature_dirs() -> list[Path]:
    return [
        p for p in _FEATURES_DIR.glob("*/*") if p.is_dir() and p.name != "__pycache__"
    ]


def test_feature_ids_are_unique_across_domains():
    seen: dict[str, Path] = {}
    for feature_dir in _feature_dirs():
        feature_id = feature_dir.name
        assert feature_id not in seen, (
            f"Duplicate feature id {feature_id!r}: {seen[feature_id]} and "
            f"{feature_dir}. Feature directory names are the flat registry id and "
            f"must be unique across ai/features/."
        )
        seen[feature_id] = feature_dir


def test_moved_prompt_feature_leaves_no_legacy_copy():
    for feature_dir in _feature_dirs():
        if not (feature_dir / "prompts").is_dir():
            continue
        legacy = _LEGACY_PROMPT_DEFS / feature_dir.name
        assert not legacy.exists(), (
            f"{feature_dir.name} moved to {feature_dir} but a legacy copy remains at "
            f"{legacy}; remove it so the prompt id resolves from one place."
        )
