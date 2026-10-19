# pylint: disable=file-naming-for-tests
"""Regression tests for DCR (Duo Code Review) prompt definitions.

Prompt caching only pays off on nodes that loop and read back their own prefix. One-shot nodes pay the write premium
and never read it, so they must keep ``cache_control_injection_points: []``; looping nodes must leave the key absent so
the default breakpoints apply. These tests catch a new version copied from an older template that gets this wrong in
either direction.
"""

from pathlib import Path

import pytest
import yaml

_PROMPTS_DEFINITIONS_DIR = (
    Path(__file__).parent.parent.parent / "ai_gateway" / "prompts" / "definitions"
)

# Whether caching must stay disabled for each DCR prompt directory.
# See https://gitlab.com/gitlab-org/gitlab/-/work_items/630239
_DCR_PROMPT_CACHING_DISABLED = {
    "analyze_prescan_codebase_results": True,  # never loops
    "review_merge_request_dap": True,  # almost never loops
    "code_review_prescan": False,  # loops
    "explore_directories_for_prescan": False,  # loops
}


def _collect_dcr_base_yaml_files() -> list[tuple[Path, bool]]:
    """Return each base YAML file of every DCR prompt directory, paired with whether caching must stay disabled."""
    files: list[tuple[Path, bool]] = []
    for prompt_dir, caching_disabled in _DCR_PROMPT_CACHING_DISABLED.items():
        base_dir = _PROMPTS_DEFINITIONS_DIR / prompt_dir / "base"
        files.extend((f, caching_disabled) for f in sorted(base_dir.glob("*.yml")))
    return files


_DCR_BASE_YAML_FILES = _collect_dcr_base_yaml_files()


def test_every_dcr_dir_has_base_yaml() -> None:
    """Guard against silent test-skip when a DCR prompt directory is renamed or moved.

    ``pytest.mark.parametrize`` with an empty list silently skips all tests, which would
    allow regressions to go undetected.  This test fails loudly if any expected directory
    is missing or contains no YAML files.
    """
    for prompt_dir in _DCR_PROMPT_CACHING_DISABLED:
        base_dir = _PROMPTS_DEFINITIONS_DIR / prompt_dir / "base"
        assert list(base_dir.glob("*.yml")), f"no base YAML under {prompt_dir}/base"


@pytest.mark.parametrize(
    ("yaml_file", "caching_disabled"),
    _DCR_BASE_YAML_FILES,
    ids=[
        f.relative_to(_PROMPTS_DEFINITIONS_DIR).as_posix()
        for f, _ in _DCR_BASE_YAML_FILES
    ],
)
def test_dcr_prompt_cache_control_injection_points(
    yaml_file: Path, caching_disabled: bool
) -> None:
    """One-shot DCR prompts must set ``cache_control_injection_points: []``; looping ones must omit the key."""
    content = yaml.safe_load(yaml_file.read_text())
    params = content.get("params", {})
    relative_path = yaml_file.relative_to(_PROMPTS_DEFINITIONS_DIR)

    if caching_disabled:
        assert params.get("cache_control_injection_points") == [], (
            f"{relative_path} must have 'params.cache_control_injection_points: []' "
            "to disable prompt caching on a one-shot DCR node"
        )
    else:
        assert "cache_control_injection_points" not in params, (
            f"{relative_path} must not set 'params.cache_control_injection_points' "
            "so the default cache breakpoints apply on a looping DCR node"
        )
