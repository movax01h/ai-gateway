# pylint: disable=file-naming-for-tests
"""Structural guard: ids under ai/features/ and ai/shared/ must be unique.

A feature's directory name is its id in the prompt and flow-config registries, an
ai/shared/ directory name is a prompt id, and each entry in a feature's
prompts/subprompts_index.yml is a prompt id. Two sources sharing an id would fail at
boot, so this catches the collision statically in CI.

The ``test_*`` functions without arguments sweep the real tree on purpose: they guard
what is checked in. The others run the same checks on synthetic trees.
"""

from pathlib import Path

from lib.subprompts_index import SUBPROMPTS_INDEX, read_subprompts_index

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FEATURES_DIR = _REPO_ROOT / "ai" / "features"
_SHARED_DIR = _REPO_ROOT / "ai" / "shared"
_LEGACY_PROMPT_DEFS = _REPO_ROOT / "ai_gateway" / "prompts" / "definitions"


def _dirs(pattern: str, root: Path) -> list[Path]:
    return [p for p in root.glob(pattern) if p.is_dir() and p.name != "__pycache__"]


def _listed_prompts(features_dir: Path) -> list[tuple[str, Path]]:
    return [
        listed
        for feature_dir in _dirs("*/*", features_dir)
        for listed in read_subprompts_index(feature_dir / "prompts").items()
    ]


def _duplicate_ids(features_dir: Path, shared_dir: Path) -> list[str]:
    """Return one message per id that two directories or index entries share."""
    ids = [
        (d.name, d) for d in _dirs("*/*", features_dir) + _dirs("*", shared_dir)
    ] + _listed_prompts(features_dir)
    seen: dict[str, Path] = {}
    duplicates = []
    for registry_id, directory in ids:
        if registry_id in seen:
            duplicates.append(f"{registry_id!r}: {seen[registry_id]} and {directory}")
        seen.setdefault(registry_id, directory)
    return duplicates


def _legacy_shadows(
    features_dir: Path, shared_dir: Path, legacy_dir: Path
) -> list[str]:
    """Return one message per registered prompt id that also exists in the legacy root."""
    moved = (
        [
            (d, [d.name, f"{d.parent.name}/{d.name}"])
            for d in _dirs("*/*", features_dir)
            if (d / "prompts").is_dir()
        ]
        + [(d, [d.name]) for d in _dirs("*", shared_dir) if (d / "prompts").is_dir()]
        + [(d, [prompt_id]) for prompt_id, d in _listed_prompts(features_dir)]
    )
    return [
        f"{prompt_id!r}: {directory} and {legacy_dir / prompt_id}"
        for directory, prompt_ids in moved
        for prompt_id in prompt_ids
        if (legacy_dir / prompt_id).exists()
    ]


def _unlisted_prompt_dirs(features_dir: Path) -> list[Path]:
    """Return each ``prompts/<name>/`` dir with a ``base/<version>.yml`` that no index lists."""
    prompt_dirs = sorted(
        {
            version.parent.parent
            for version in features_dir.glob("*/*/prompts/*/base/*.yml")
        }
    )
    return [
        prompt_dir
        for prompt_dir in prompt_dirs
        if prompt_dir not in read_subprompts_index(prompt_dir.parent).values()
    ]


def _write_listed_prompt(prompts: Path, prompt_id: str, directory: str) -> None:
    (prompts / directory / "base").mkdir(parents=True)
    (prompts / directory / "base" / "1.0.0.yml").write_text("name: x\n")
    (prompts / SUBPROMPTS_INDEX).write_text(f"{prompt_id}: {directory}\n")


def test_feature_ids_are_unique_across_domains():
    duplicates = _duplicate_ids(_FEATURES_DIR, _SHARED_DIR)
    assert not duplicates, (
        f"Duplicate ids: {duplicates}. Feature and shared directory names, and the "
        f"prompt ids in each {SUBPROMPTS_INDEX}, are registry ids and must be unique "
        f"across ai/features/ and ai/shared/."
    )


def test_registered_prompt_ids_do_not_shadow_legacy_prompts():
    shadows = _legacy_shadows(_FEATURES_DIR, _SHARED_DIR, _LEGACY_PROMPT_DEFS)
    assert not shadows, (
        f"These prompt ids are registered and also exist in the legacy root: "
        f"{shadows}. The registered copy hides the legacy one. Remove the legacy copy "
        f"after a move, or rename one of the two prompts."
    )


def test_every_prompt_dir_with_base_is_listed_in_its_index():
    unlisted = _unlisted_prompt_dirs(_FEATURES_DIR)
    assert not unlisted, (
        f"These directories hold base/<version>.yml but no {SUBPROMPTS_INDEX} lists "
        f"them, so they register no prompt: "
        f"{[str(d.relative_to(_REPO_ROOT)) for d in unlisted]}. Add "
        f"'<prompt_id>: <directory>' to prompts/{SUBPROMPTS_INDEX}, or move the "
        f"version files out of base/."
    )


def test_index_id_colliding_with_a_feature_dir_is_a_duplicate(tmp_path: Path):
    features, shared = tmp_path / "features", tmp_path / "shared"
    (features / "cli" / "my_flow_fetch").mkdir(parents=True)
    _write_listed_prompt(
        features / "insights" / "my_flow" / "prompts", "my_flow_fetch", "fetch"
    )

    assert _duplicate_ids(features, shared) == [
        f"'my_flow_fetch': {features / 'cli' / 'my_flow_fetch'} and "
        f"{features / 'insights' / 'my_flow' / 'prompts' / 'fetch'}"
    ]


def test_index_ids_colliding_across_features_are_duplicates(tmp_path: Path):
    features = tmp_path / "features"
    _write_listed_prompt(features / "a" / "one" / "prompts", "shared_id", "x")
    _write_listed_prompt(features / "b" / "two" / "prompts", "shared_id", "y")

    assert len(_duplicate_ids(features, tmp_path / "shared")) == 1


def test_index_id_in_the_legacy_root_is_a_shadow(tmp_path: Path):
    features, legacy = tmp_path / "features", tmp_path / "definitions"
    prompts = features / "insights" / "my_flow" / "prompts"
    _write_listed_prompt(prompts, "my_flow_fetch", "fetch")
    (legacy / "my_flow_fetch" / "base").mkdir(parents=True)

    assert _legacy_shadows(features, tmp_path / "shared", legacy) == [
        f"'my_flow_fetch': {prompts / 'fetch'} and {legacy / 'my_flow_fetch'}"
    ]


def test_unlisted_prompt_dirs_finds_base_versions_without_an_entry(tmp_path: Path):
    prompts = tmp_path / "insights" / "my_flow" / "prompts"
    _write_listed_prompt(prompts, "my_flow_fetch", "fetch")
    (prompts / "common" / "base").mkdir(parents=True)
    (prompts / "common" / "base" / "1.0.0.yml").write_text("name: x\n")
    (prompts / "partials" / "base").mkdir(parents=True)
    (prompts / "partials" / "base" / "1.0.0.jinja").write_text("include only")
    (prompts / "base").mkdir()
    (prompts / "base" / "1.0.0.yml").write_text("name: x\n")
    stray = tmp_path / "cli" / "no_index" / "prompts" / "stray"
    (stray / "base").mkdir(parents=True)
    (stray / "base" / "1.0.0.yml").write_text("name: x\n")

    assert _unlisted_prompt_dirs(tmp_path) == [stray, prompts / "common"]
