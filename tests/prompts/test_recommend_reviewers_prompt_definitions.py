# pylint: disable=file-naming-for-tests
"""Regression tests for recommend_reviewers prompt definitions.

Each run is one LLM call ending in per-run reviewer data, so caching only adds writes. Guards against a new version
copied from an older template dropping the opt-out.
"""

from pathlib import Path

import pytest
import yaml

_PROMPTS_DEFINITIONS_DIR = (
    Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
)

_RECOMMEND_REVIEWERS_PROMPT_DIRS = [
    "recommend_reviewers_assign",
    "recommend_reviewers_post",
]


def _collect_base_yaml_files() -> list[Path]:
    files: list[Path] = []
    for prompt_dir in _RECOMMEND_REVIEWERS_PROMPT_DIRS:
        base_dir = _PROMPTS_DEFINITIONS_DIR / prompt_dir / "base"
        files.extend(sorted(base_dir.glob("*.yml")))
    return files


_BASE_YAML_FILES = _collect_base_yaml_files()


def test_every_recommend_reviewers_dir_has_base_yaml() -> None:
    """Fail loudly instead of silently skipping the parametrized test if a directory moves."""
    for prompt_dir in _RECOMMEND_REVIEWERS_PROMPT_DIRS:
        base_dir = _PROMPTS_DEFINITIONS_DIR / prompt_dir / "base"
        assert list(base_dir.glob("*.yml")), f"no base YAML under {prompt_dir}/base"


@pytest.mark.parametrize(
    "yaml_file",
    _BASE_YAML_FILES,
    ids=[f.relative_to(_PROMPTS_DEFINITIONS_DIR).as_posix() for f in _BASE_YAML_FILES],
)
def test_recommend_reviewers_prompt_has_cache_control_injection_points_disabled(
    yaml_file: Path,
) -> None:
    """Every recommend_reviewers base prompt must set ``cache_control_injection_points: []``."""
    content = yaml.safe_load(yaml_file.read_text())
    params = content.get("params", {})
    assert params.get("cache_control_injection_points") == [], (
        f"{yaml_file.relative_to(_PROMPTS_DEFINITIONS_DIR)} must have "
        "'params.cache_control_injection_points: []' to disable prompt caching"
    )
