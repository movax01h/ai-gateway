# pylint: disable=file-naming-for-tests
"""Regression tests for DCR (Duo Code Review) prompt definitions.

These tests guard against a class of regression where a new prompt version is
created by copying an older template and the developer forgets to carry
``cache_control_injection_points: []`` forward.  Without the explicit empty
list the default caching behaviour kicks in and silently re-enables prompt
caching for DCR prompts, which must remain disabled.
"""

from pathlib import Path

import pytest
import yaml

_PROMPTS_DEFINITIONS_DIR = (
    Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
)

# The four DCR feature-set prompt directories that must never use prompt caching.
_DCR_PROMPT_DIRS = [
    "explore_directories_for_prescan",
    "code_review_prescan",
    "analyze_prescan_codebase_results",
    "review_merge_request_dap",
]


def _collect_dcr_base_yaml_files() -> list[Path]:
    """Return all YAML files found under the ``base`` sub-directory of each DCR prompt directory."""
    files: list[Path] = []
    for prompt_dir in _DCR_PROMPT_DIRS:
        base_dir = _PROMPTS_DEFINITIONS_DIR / prompt_dir / "base"
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
        base_dir = _PROMPTS_DEFINITIONS_DIR / prompt_dir / "base"
        assert list(base_dir.glob("*.yml")), f"no base YAML under {prompt_dir}/base"


@pytest.mark.parametrize(
    "yaml_file",
    _DCR_BASE_YAML_FILES,
    ids=[
        f.relative_to(_PROMPTS_DEFINITIONS_DIR).as_posix() for f in _DCR_BASE_YAML_FILES
    ],
)
def test_dcr_prompt_has_cache_control_injection_points_disabled(
    yaml_file: Path,
) -> None:
    """Every base prompt definition in the DCR feature set must explicitly set ``cache_control_injection_points: []`` to
    prevent prompt caching.

    This catches regressions where a new version is created by copying an older template without carrying the setting
    forward.
    """
    content = yaml.safe_load(yaml_file.read_text())
    params = content.get("params", {})
    assert params.get("cache_control_injection_points") == [], (
        f"{yaml_file.relative_to(_PROMPTS_DEFINITIONS_DIR)} must have "
        "'params.cache_control_injection_points: []' to disable prompt caching for DCR prompts"
    )
