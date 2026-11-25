# pylint: disable=file-naming-for-tests
"""Regression tests for DCR (Duo Code Review) prompt definitions.

Prompt caching costs more than it saves on every DCR node: each node runs once or twice
per review, so the cache write premium is rarely paid back by reads. Every base prompt
must set ``cache_control_injection_points: []``, because leaving the key out applies the
default breakpoints. These tests catch a new version copied from an older template that
drops the setting.
"""

from pathlib import Path

import pytest
import yaml

_REVIEW_FEATURES_DIR = Path(__file__).parent.parent.parent

_DCR_PROMPT_DIRS = [
    "analyze_prescan_codebase_results",
    "code_review_prescan",
    "explore_directories_for_prescan",
    "review_merge_request_dap",
]


def _collect_dcr_base_yaml_files() -> list[Path]:
    """Return every base YAML file of every DCR prompt directory."""
    files: list[Path] = []
    for prompt_dir in _DCR_PROMPT_DIRS:
        base_dir = _REVIEW_FEATURES_DIR / prompt_dir / "prompts" / "base"
        files.extend(sorted(base_dir.glob("*.yml")))
    return files


_DCR_BASE_YAML_FILES = _collect_dcr_base_yaml_files()


def test_every_dcr_dir_has_base_yaml() -> None:
    """Guard against silent test-skip when a DCR prompt directory is renamed or moved.

    ``pytest.mark.parametrize`` with an empty list silently skips all tests, which would
    allow regressions to go undetected.  This test fails loudly if any expected directory
    is missing or contains no YAML files.
    """
    for prompt_dir in _DCR_PROMPT_DIRS:
        base_dir = _REVIEW_FEATURES_DIR / prompt_dir / "prompts" / "base"
        assert list(base_dir.glob("*.yml")), f"no base YAML under {prompt_dir}/base"


@pytest.mark.parametrize(
    "yaml_file",
    _DCR_BASE_YAML_FILES,
    ids=[f.relative_to(_REVIEW_FEATURES_DIR).as_posix() for f in _DCR_BASE_YAML_FILES],
)
def test_dcr_prompt_has_cache_control_injection_points_disabled(
    yaml_file: Path,
) -> None:
    """Every DCR base prompt must set ``cache_control_injection_points: []``."""
    content = yaml.safe_load(yaml_file.read_text())
    params = content.get("params", {})

    assert params.get("cache_control_injection_points") == [], (
        f"{yaml_file.relative_to(_REVIEW_FEATURES_DIR)} must have "
        "'params.cache_control_injection_points: []' to disable prompt caching"
    )
