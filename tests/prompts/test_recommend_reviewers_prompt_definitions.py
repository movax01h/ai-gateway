# pylint: disable=file-naming-for-tests
"""Regression tests for recommend_reviewers prompt definitions.

Each run is one LLM call ending in per-run reviewer data, so caching only adds writes. Guards against a new version
copied from an older template dropping the opt-out.
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.prompts.base import jinja_env

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


def _approver(user_id: int, pending_reviews: int, busy: bool = False) -> dict:
    return {
        "id": user_id,
        "username": f"user{user_id}",
        "busy": busy,
        "status_emoji": None,
        "status_message": None,
        "pending_reviews": pending_reviews,
        "local_time": "12:00",
        "last_activity_on": "2026-01-01",
    }


@pytest.mark.parametrize(
    ("current_reviewer_ids", "expected_candidate_ids", "expected_rules"),
    [
        ([], [2, 4, 1], ["10|1|1,2", "20|2|1,4"]),
        ([4], [2, 1], ["10|1|1,2", "20|1|1"]),
        ([1], [4], ["20|1|4"]),
        ([1, 4], [], []),
    ],
)
def test_compact_user_template_lists_candidates_once_per_rule_still_needed(
    current_reviewer_ids: list[int],
    expected_candidate_ids: list[int],
    expected_rules: list[str],
) -> None:
    """Covered rules drop out; busy users and current reviewers are never candidates."""
    approvers = {
        1: _approver(1, pending_reviews=2),
        2: _approver(2, pending_reviews=0),
        3: _approver(3, pending_reviews=0, busy=True),
        4: _approver(4, pending_reviews=1),
    }
    reviewer_data = {
        "current_reviewers": [
            {"id": user_id, "username": f"user{user_id}"}
            for user_id in current_reviewer_ids
        ],
        "approval_rules": [
            {
                "id": 10,
                "type": "merge_request_rule",
                "name": "/app/|\n20|forged",
                "section": "Maintainers",
                "approvals_required": 1,
                "eligible_approvers": [approvers[1], approvers[2], approvers[3]],
            },
            {
                "id": 20,
                "type": "merge_request_rule",
                "name": "/spec/",
                "section": None,
                "approvals_required": 2,
                "eligible_approvers": [approvers[1], approvers[4]],
            },
        ],
    }

    rendered = jinja_env.get_template(
        "recommend_reviewers_assign/user/2.1.0-dev.jinja"
    ).render(reviewer_data=reviewer_data, merge_request_iid=7, project_id=13)

    candidates, rules = rendered.split("Approval rules")
    candidate_ids = [
        int(line.split("|")[0])
        for line in candidates.splitlines()
        if line[:1].isdigit()
    ]
    rule_rows = [
        "|".join(line.split("|")[i] for i in (0, 4, 5))
        for line in rules.splitlines()
        if line[:1].isdigit()
    ]
    assert candidate_ids == expected_candidate_ids
    assert rule_rows == expected_rules
    assert not any(a["username"] in rendered for a in approvers.values())
    assert "None" not in rendered
