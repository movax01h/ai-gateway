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


def _hides_registered_files(legacy: Path, served: Path) -> bool:
    """Return whether a legacy dir would win over files that a registered root serves.

    Two things hide registered files. ``PackageLoader`` runs before ``FeatureRootLoader``, so a legacy include fragment
    that a registered root also serves under the same name hides the registered copy. And once an id is registered,
    ``LocalPromptRegistry`` resolves it from the registered root only, with no fallback to the legacy dir, so any legacy
    prompt version (``.yml``) becomes unreachable even if the registered root has no file of that name. A legacy dir
    that holds neither, for example a ``common/`` that still holds fragments which have not moved yet, hides nothing:
    those includes fall through to the registered root.
    """
    if not legacy.is_dir():
        return False
    return any(
        legacy_file.suffix == ".yml"
        or (served / legacy_file.relative_to(legacy)).is_file()
        for legacy_file in legacy.rglob("*")
        if legacy_file.is_file() and "__pycache__" not in legacy_file.parts
    )


def _legacy_shadows(
    features_dir: Path, shared_dir: Path, legacy_dir: Path
) -> list[str]:
    """Return one message per registered prompt id whose legacy dir hides a registered file."""
    moved = (
        [
            (d, d / "prompts", [d.name, f"{d.parent.name}/{d.name}"])
            for d in _dirs("*/*", features_dir)
            if (d / "prompts").is_dir()
        ]
        + [
            (d, d / "prompts", [d.name])
            for d in _dirs("*", shared_dir)
            if (d / "prompts").is_dir()
        ]
        + [(d, d, [prompt_id]) for prompt_id, d in _listed_prompts(features_dir)]
    )
    return [
        f"{prompt_id!r}: {directory} and {legacy_dir / prompt_id}"
        for directory, served, prompt_ids in moved
        for prompt_id in prompt_ids
        if _hides_registered_files(legacy_dir / prompt_id, served)
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
    (legacy / "my_flow_fetch" / "base" / "1.0.0.yml").write_text("name: x\n")

    assert _legacy_shadows(features, tmp_path / "shared", legacy) == [
        f"'my_flow_fetch': {prompts / 'fetch'} and {legacy / 'my_flow_fetch'}"
    ]


def test_include_only_legacy_dir_with_no_overlap_is_not_a_shadow(tmp_path: Path):
    features, legacy = tmp_path / "features", tmp_path / "definitions"
    _write_listed_prompt(
        features / "insights" / "my_flow" / "prompts", "my_flow_fetch", "fetch"
    )
    (legacy / "my_flow_fetch" / "partials").mkdir(parents=True)
    (legacy / "my_flow_fetch" / "partials" / "1.0.0.jinja").write_text("include only")

    assert not _legacy_shadows(features, tmp_path / "shared", legacy)


def test_legacy_include_fragment_served_by_a_shared_dir_is_a_shadow(tmp_path: Path):
    shared, legacy = tmp_path / "shared", tmp_path / "definitions"
    served = shared / "common" / "prompts" / "push_retry"
    served.mkdir(parents=True)
    (served / "1.0.0.jinja").write_text("moved")
    (legacy / "common" / "push_retry").mkdir(parents=True)
    (legacy / "common" / "push_retry" / "1.0.0.jinja").write_text("stale")

    assert _legacy_shadows(tmp_path / "features", shared, legacy) == [
        f"'common': {shared / 'common'} and {legacy / 'common'}"
    ]


def test_legacy_prompt_version_without_a_registered_counterpart_is_a_shadow(
    tmp_path: Path,
):
    features, legacy = tmp_path / "features", tmp_path / "definitions"
    prompts = features / "insights" / "my_flow" / "prompts"
    _write_listed_prompt(prompts, "my_flow_fetch", "fetch")
    (legacy / "my_flow_fetch" / "gpt").mkdir(parents=True)
    (legacy / "my_flow_fetch" / "gpt" / "1.0.0.yml").write_text("name: x\n")

    assert len(_legacy_shadows(features, tmp_path / "shared", legacy)) == 1


def test_legacy_jinja_only_dir_served_by_a_shared_dir_is_a_shadow(tmp_path: Path):
    shared, legacy = tmp_path / "shared", tmp_path / "definitions"
    (shared / "agents_dot_md" / "prompts" / "base").mkdir(parents=True)
    (shared / "agents_dot_md" / "prompts" / "base" / "1.0.0.jinja").write_text("moved")
    (legacy / "agents_dot_md" / "base").mkdir(parents=True)
    (legacy / "agents_dot_md" / "base" / "1.0.0.jinja").write_text("stale")

    assert len(_legacy_shadows(tmp_path / "features", shared, legacy)) == 1


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
