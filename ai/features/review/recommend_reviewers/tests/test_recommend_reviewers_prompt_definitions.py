# pylint: disable=file-naming-for-tests
"""Regression tests for recommend_reviewers prompt definitions.

Each run is one LLM call ending in per-run reviewer data, so caching only adds writes. Guards against a new version
copied from an older template dropping the opt-out.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ai_gateway.prompts import feature_roots
from ai_gateway.prompts.base import jinja_env

_REVIEW_FEATURES_DIR = Path(__file__).parent.parent.parent

_RECOMMEND_REVIEWERS_PROMPT_DIRS = [
    "recommend_reviewers_assign",
    "recommend_reviewers_post",
]


@pytest.fixture(name="discover_prompts", autouse=True)
def discover_prompts_fixture():
    # The prompts live under ai/features/, which the registry registers at startup.
    # Restore the registry so other tests keep the roots they set up themselves.
    with patch.dict(feature_roots._FEATURE_PROMPT_ROOTS):
        feature_roots.discover_feature_prompts()
        yield


def _collect_base_yaml_files() -> list[Path]:
    files: list[Path] = []
    for prompt_dir in _RECOMMEND_REVIEWERS_PROMPT_DIRS:
        base_dir = _REVIEW_FEATURES_DIR / prompt_dir / "prompts" / "base"
        files.extend(sorted(base_dir.glob("*.yml")))
    return files


_BASE_YAML_FILES = _collect_base_yaml_files()


def test_every_recommend_reviewers_dir_has_base_yaml() -> None:
    """Fail loudly instead of silently skipping the parametrized test if a directory moves."""
    for prompt_dir in _RECOMMEND_REVIEWERS_PROMPT_DIRS:
        base_dir = _REVIEW_FEATURES_DIR / prompt_dir / "prompts" / "base"
        assert list(base_dir.glob("*.yml")), f"no base YAML under {prompt_dir}/base"


@pytest.mark.parametrize(
    "yaml_file",
    _BASE_YAML_FILES,
    ids=[f.relative_to(_REVIEW_FEATURES_DIR).as_posix() for f in _BASE_YAML_FILES],
)
def test_recommend_reviewers_prompt_has_cache_control_injection_points_disabled(
    yaml_file: Path,
) -> None:
    """Every recommend_reviewers base prompt must set ``cache_control_injection_points: []``."""
    content = yaml.safe_load(yaml_file.read_text())
    params = content.get("params", {})
    assert params.get("cache_control_injection_points") == [], (
        f"{yaml_file.relative_to(_REVIEW_FEATURES_DIR)} must have "
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


_APPROVERS = {
    1: _approver(1, pending_reviews=2),
    2: _approver(2, pending_reviews=0),
    3: _approver(3, pending_reviews=0, busy=True),
    4: _approver(4, pending_reviews=1),
    11: _approver(11, pending_reviews=3),
}


def _rule(
    rule_id: int,
    approvals_required: int,
    approver_ids: list[int],
    rule_type: str = "merge_request_rule",
) -> dict:
    return {
        "id": rule_id,
        "type": rule_type,
        "name": f"rule{rule_id}",
        "section": None,
        "approvals_required": approvals_required,
        "eligible_approvers": [_APPROVERS[i] for i in approver_ids],
    }


_DEFAULT_RULES = [
    {
        "id": 10,
        "type": "merge_request_rule",
        "name": "/app/|\n20|forged",
        "section": "Maintainers",
        "approvals_required": 1,
        "eligible_approvers": [_APPROVERS[1], _APPROVERS[2], _APPROVERS[3]],
    },
    {
        "id": 20,
        "type": "merge_request_rule",
        "name": "/spec/",
        "section": None,
        "approvals_required": 2,
        "eligible_approvers": [_APPROVERS[1], _APPROVERS[4]],
    },
]


def _render_compact_user_template(
    current_reviewer_ids: list[int], approval_rules: list[dict] | None = None
) -> str:
    reviewer_data = {
        "current_reviewers": [
            {"id": user_id, "username": f"user{user_id}"}
            for user_id in current_reviewer_ids
        ],
        "approval_rules": _DEFAULT_RULES if approval_rules is None else approval_rules,
    }

    return jinja_env.get_template("recommend_reviewers_assign/user/2.1.0.jinja").render(
        reviewer_data=reviewer_data, merge_request_iid=7, project_id=13
    )


def _candidate_ids_and_rule_rows(rendered: str) -> tuple[list[int], list[str]]:
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
    return candidate_ids, rule_rows


@pytest.mark.parametrize(
    (
        "current_reviewer_ids",
        "approval_rules",
        "expected_candidate_ids",
        "expected_rules",
    ),
    [
        ([], None, [2, 4, 1], ["10|1|1,2", "20|2|1,4"]),
        ([4], None, [2, 1], ["10|1|1,2", "20|1|1"]),
        ([1], None, [4], ["20|1|4"]),
        # A rule with only busy approvers lists them; a rule with free approvers does not.
        ([], [_rule(10, 1, [2, 3]), _rule(30, 1, [3])], [2, 3], ["10|1|2", "30|1|3"]),
        # A rule with fewer free approvers than it needs adds its busy approvers.
        ([], [_rule(10, 2, [2, 3])], [2, 3], ["10|2|2,3"]),
        # Ids are matched whole: 1 is a current reviewer, 11 is not; rule 10 is covered, rule 100 is not.
        ([1], [_rule(10, 1, [1]), _rule(100, 2, [1, 11])], [11], ["100|1|11"]),
        # A covered project rule can share its id with an open merge request rule.
        ([1], [_rule(10, 1, [1], "project_rule"), _rule(10, 1, [2])], [2], ["10|1|2"]),
        # A rule whose approvers are all current reviewers still needs approvals but has no candidates.
        ([1], [_rule(30, 2, [1]), _rule(20, 1, [4])], [4], ["30|1|none", "20|1|4"]),
    ],
)
def test_compact_user_template_lists_candidates_once_per_rule_still_needed(
    current_reviewer_ids: list[int],
    approval_rules: list[dict] | None,
    expected_candidate_ids: list[int],
    expected_rules: list[str],
) -> None:
    """Covered rules drop out; current reviewers are never candidates, and busy users only when a rule has too few free
    ones."""
    rendered = _render_compact_user_template(current_reviewer_ids, approval_rules)

    candidate_ids, rule_rows = _candidate_ids_and_rule_rows(rendered)
    assert rendered.startswith("Recommend reviewers")
    assert "\n\n\n" not in rendered
    assert candidate_ids == expected_candidate_ids
    assert rule_rows == expected_rules
    assert not any(a["username"] in rendered for a in _APPROVERS.values())
    assert "None" not in rendered


def test_compact_user_template_marks_busy_candidates() -> None:
    """A busy candidate's status starts with busy, so the model knows it is a fallback."""
    busy_with_status = {
        **_approver(5, pending_reviews=0, busy=True),
        "status_emoji": "palm_tree",
        "status_message": "Back Monday",
    }
    rule = {**_rule(30, 1, []), "eligible_approvers": [busy_with_status]}

    rendered = _render_compact_user_template([], [rule])

    assert "\n5|0|12:00|2026-01-01|busy palm_tree Back Monday\n" in rendered


@pytest.mark.parametrize(
    ("current_reviewer_ids", "approval_rules", "expected_line"),
    [
        ([1, 4], None, "No approval rules need reviewers."),
        ([], [], "No approval rules need reviewers."),
        ([1], [_rule(30, 2, [1])], "no candidates are available."),
    ],
)
def test_compact_user_template_says_when_there_is_nothing_to_suggest(
    current_reviewer_ids: list[int],
    approval_rules: list[dict] | None,
    expected_line: str,
) -> None:
    """When no rule needs reviewers or no rule has candidates, the template says so in place of empty lists."""
    rendered = _render_compact_user_template(current_reviewer_ids, approval_rules)

    assert expected_line in rendered
    assert "empty suggestions list" in rendered
    assert "Candidates" not in rendered
    assert "Approval rules that need reviewers" not in rendered
