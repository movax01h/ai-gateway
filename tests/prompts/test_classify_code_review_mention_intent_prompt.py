# pylint: disable=file-naming-for-tests
"""Tests for the user template of classify_code_review_mention_intent 2.0.0.

Version 2.0.0 gives the classifier the review context of the mention, so that a reply such as "please approve" in a
thread that a Duo review started goes to a new review. See
https://gitlab.com/gitlab-org/gitlab/-/work_items/635111.
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.prompts.base import jinja_env
from ai_gateway.prompts.feature_roots import discover_feature_prompts

_TEMPLATE = "classify_code_review_mention_intent/user/2.0.0.jinja"

_NO_CONTEXT = {
    "message": "please approve",
    "thread": "",
    "thread_started_by_duo_review": False,
    "duo_review_state": "not_reviewer",
}


@pytest.fixture(scope="module", autouse=True)
def _feature_prompts():
    discover_feature_prompts()


def _render(**overrides) -> str:
    return jinja_env.get_template(_TEMPLATE).render(**{**_NO_CONTEXT, **overrides})


def test_renders_the_message_after_the_context():
    rendered = _render()

    assert rendered.index("</context>") < rendered.index("Message: please approve")


@pytest.mark.parametrize(
    ("duo_review_state", "expected"),
    [
        (None, "no"),
        ("not_reviewer", "no"),
        ("unreviewed", "yes"),
        ("review_started", "yes"),
        ("reviewed", "yes"),
        ("requested_changes", "yes"),
        ("approved", "yes"),
        ("unapproved", "yes"),
    ],
)
def test_duo_reviewer(duo_review_state, expected):
    rendered = _render(duo_review_state=duo_review_state)

    assert f"Duo is a reviewer of this merge request: {expected}" in rendered


def test_without_a_duo_review_thread():
    rendered = _render()

    assert "Thread started by a Duo review: no" in rendered
    assert "Earlier messages in this thread" not in rendered


def test_reply_in_a_duo_review_thread():
    thread = '<message author="GitLabDuo">\nOne minor finding.\n</message>'

    rendered = _render(
        thread=thread,
        thread_started_by_duo_review=True,
        duo_review_state="requested_changes",
    )

    assert "Thread started by a Duo review: yes" in rendered
    assert f"Earlier messages in this thread, oldest first:\n{thread}" in rendered


_PROMPT_DIR = (
    Path(__file__).parent.parent.parent
    / "ai"
    / "features"
    / "review"
    / "classify_code_review_mention_intent"
    / "prompts"
)
_BASE_DEFINITIONS = sorted((_PROMPT_DIR / "base").glob("*.yml"))


def test_base_definitions_are_discovered():
    # Guards the parametrized test below against silently matching no versions.
    assert {path.stem for path in _BASE_DEFINITIONS} >= {"1.0.0", "2.0.0"}


@pytest.mark.parametrize(
    "definition", _BASE_DEFINITIONS, ids=[path.stem for path in _BASE_DEFINITIONS]
)
def test_max_tokens_leaves_room_to_classify(definition):
    """A too-low limit truncates the classification, so the mention is routed to legacy chat without tool access.

    See https://gitlab.com/gitlab-org/gitlab/-/work_items/639435.
    """
    config = yaml.safe_load(definition.read_text())

    assert config["model"]["params"]["max_tokens"] == 1024
