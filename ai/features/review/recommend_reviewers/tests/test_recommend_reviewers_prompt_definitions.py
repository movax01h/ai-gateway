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


def _render_user_template(version: str, reviewer_data: dict) -> str:
    return jinja_env.get_template(
        f"recommend_reviewers_assign/user/{version}.jinja"
    ).render(reviewer_data=reviewer_data, merge_request_iid=7, project_id=13)


def _reviewer_data(
    current_reviewer_ids: list[int], approval_rules: list[dict] | None = None
) -> dict:
    return {
        "current_reviewers": [
            {"id": user_id, "username": f"user{user_id}"}
            for user_id in current_reviewer_ids
        ],
        "approval_rules": _DEFAULT_RULES if approval_rules is None else approval_rules,
    }


@pytest.mark.parametrize(
    ("current_reviewer_ids", "approval_rules"),
    [
        ([], None),
        ([4], None),
        ([1, 4], None),
        ([], [_rule(10, 1, [2, 3]), _rule(30, 1, [3])]),
        ([1], [_rule(30, 2, [1]), _rule(20, 1, [4])]),
        ([], []),
    ],
)
def test_user_template_3_0_0_renders_like_2_1_0_without_the_new_fields(
    current_reviewer_ids: list[int], approval_rules: list[dict] | None
) -> None:
    """Rails sends the new fields and signals behind flags, so without them nothing changes."""
    reviewer_data = _reviewer_data(current_reviewer_ids, approval_rules)

    assert _render_user_template("3.0.0", reviewer_data) == _render_user_template(
        "2.1.0", reviewer_data
    )


def _with(approver_id: int, **fields) -> dict:
    return {**_APPROVERS[approver_id], **fields}


def test_user_template_3_0_0_lists_cost_sorted_by_cost() -> None:
    """A candidate with fewer pending reviews can cost more, so candidates are listed by cost when Rails sends it."""
    rule = {
        **_rule(10, 1, []),
        "eligible_approvers": [_with(2, cost=10.0), _with(4, cost=6.0)],
    }

    rendered = _render_user_template("3.0.0", _reviewer_data([], [rule]))

    assert (
        "Candidates (id|pending_reviews|cost|local_time|last_activity_on|status):\n"
        "4|1|6.0|12:00|2026-01-01|\n"
        "2|0|10.0|12:00|2026-01-01|\n"
    ) in rendered


def test_user_template_3_0_0_sorts_by_pending_reviews_when_some_candidates_have_no_cost() -> (
    None
):
    """Sorting by a cost that some candidates lack would fail the render, so they are listed by workload."""
    rule = {
        **_rule(10, 1, []),
        "eligible_approvers": [_with(2, cost=10.0), _APPROVERS[4], _with(1, cost=4.0)],
    }

    rendered = _render_user_template("3.0.0", _reviewer_data([], [rule]))

    assert (
        "Candidates (id|pending_reviews|cost|local_time|last_activity_on|status):\n"
        "2|0|10.0|12:00|2026-01-01|\n"
        "4|1||12:00|2026-01-01|\n"
        "1|2|4.0|12:00|2026-01-01|\n"
    ) in rendered


_SIGNALS = [
    {
        "name": "path_history",
        "description": "Merged MRs on the changed files.\nHigher is more.",
    },
    {"name": "recent_reviews", "description": "Review requests in the last 7 days."},
]


def test_user_template_3_0_0_lists_declared_signals_in_declared_order() -> None:
    """Rails declares signals once and sends values per candidate, so a new signal needs no gateway change."""
    rule = {
        **_rule(10, 1, []),
        "eligible_approvers": [
            _with(
                2, signals={"recent_reviews": 12, "path_history": 4, "undeclared": 9}
            ),
            _with(4, signals={"recent_reviews": 0}),
            _APPROVERS[11],
        ],
    }

    rendered = _render_user_template(
        "3.0.0", {**_reviewer_data([], [rule]), "signals": _SIGNALS}
    )

    assert (
        "Signals (name: what it measures):\n"
        "- path_history: Merged MRs on the changed files. Higher is more.\n"
        "- recent_reviews: Review requests in the last 7 days.\n"
        "\n"
        "Candidates (id|pending_reviews|path_history|recent_reviews|local_time|last_activity_on|status):\n"
        "2|0|4|12|12:00|2026-01-01|\n"
        "4|1||0|12:00|2026-01-01|\n"
        "11|3|||12:00|2026-01-01|\n"
    ) in rendered
    assert "undeclared" not in rendered


def test_user_template_3_0_0_renders_signal_values_safely() -> None:
    """Booleans read like can_merge, and a value cannot add a column or a line."""
    signals = [{"name": "flag"}, {"name": "note|x"}, {"name": "empty"}]
    rule = {
        **_rule(10, 1, []),
        "eligible_approvers": [
            _with(2, signals={"flag": True, "note|x": "a|b\nc", "empty": None}),
            _with(4, signals={"flag": False}),
        ],
    }

    rendered = _render_user_template(
        "3.0.0", {**_reviewer_data([], [rule]), "signals": signals}
    )

    assert "Candidates (id|pending_reviews|flag|note/x|empty|local_time|" in rendered
    assert "- note/x: \n" in rendered
    assert "\n2|0|yes|a/b c||12:00|" in rendered
    assert "\n4|1|no|||12:00|" in rendered


def test_user_template_3_0_0_lists_no_signals_when_nothing_needs_reviewers() -> None:
    rendered = _render_user_template(
        "3.0.0", {**_reviewer_data([], []), "signals": _SIGNALS}
    )

    assert "empty suggestions list" in rendered
    assert "Signals" not in rendered


def test_user_template_3_0_0_lists_can_merge() -> None:
    """can_merge is yes, no, or empty when Rails did not check the candidate."""
    rule = {
        **_rule(10, 1, []),
        "eligible_approvers": [
            _with(2, can_merge=True),
            _with(4, can_merge=False),
            _approver(5, pending_reviews=2),
        ],
    }

    rendered = _render_user_template("3.0.0", _reviewer_data([], [rule]))

    assert (
        "Candidates (id|pending_reviews|can_merge|local_time|last_activity_on|status):\n"
    ) in rendered
    assert "\n2|0|yes|12:00|" in rendered
    assert "\n4|1|no|12:00|" in rendered
    assert "\n5|2||12:00|" in rendered


@pytest.mark.parametrize(
    ("extra", "expected_line"),
    [
        (
            {"merger_required": True},
            "A reviewer who can merge is required: neither the author nor a current reviewer can merge",
        ),
        ({"suggested_set": [4, 2]}, "Suggested set (ids): 4,2"),
    ],
)
def test_user_template_3_0_0_adds_the_merger_and_suggested_set_lines(
    extra: dict, expected_line: str
) -> None:
    rendered = _render_user_template("3.0.0", {**_reviewer_data([]), **extra})

    assert expected_line in rendered
    assert rendered.index("Approval rules that need reviewers") < rendered.index(
        expected_line
    )


@pytest.mark.parametrize(
    "extra", [{"merger_required": False}, {"suggested_set": []}, {"signals": []}]
)
def test_user_template_3_0_0_skips_lines_for_empty_values(extra: dict) -> None:
    reviewer_data = {**_reviewer_data([]), **extra}

    assert _render_user_template("3.0.0", reviewer_data) == _render_user_template(
        "2.1.0", _reviewer_data([])
    )
