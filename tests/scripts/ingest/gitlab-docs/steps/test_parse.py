"""Tests for the GitLab docs ingestion parser."""

import importlib.util
from pathlib import Path

import pytest

PARSE_PATH = (
    Path(__file__).parents[5]
    / "scripts"
    / "ingest"
    / "gitlab-docs"
    / "steps"
    / "parse.py"
)

_spec = importlib.util.spec_from_file_location("gitlab_docs_parse", PARSE_PATH)
assert _spec is not None and _spec.loader is not None
parse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parse)

BODY = (
    "Some body text that is long enough to be kept as its own chunk by the splitter. "
    * 3
    + "\n"
)


@pytest.mark.parametrize(
    ("content", "expected_title"),
    [
        pytest.param(
            '---\nstage: Foo\ntitle: "GitLab Advanced SAST rules: Python"\n---\n\n'
            + BODY,
            "GitLab Advanced SAST rules: Python",
            id="front_matter_title_without_heading",
        ),
        pytest.param(
            "---\ntitle: Page title\n---\n\n## Prerequisites\n\n" + BODY,
            "Page title",
            id="front_matter_title_wins_over_heading",
        ),
        pytest.param(
            "---\nstage: Foo\n---\n\n# Heading title **(ULTIMATE)**\n\n" + BODY,
            "Heading title",
            id="heading_without_front_matter_title",
        ),
        pytest.param(
            "# Heading title\n\n" + BODY,
            "Heading title",
            id="heading_without_front_matter",
        ),
        pytest.param(
            "---\ntitle: [unclosed\n---\n\n# Heading title\n\n" + BODY,
            "Heading title",
            id="invalid_front_matter",
        ),
        pytest.param(
            "---\ntitle: '  '\n---\n\n# Heading title\n\n" + BODY,
            "Heading title",
            id="blank_front_matter_title",
        ),
        pytest.param(
            "---\nstage: Foo\n---\n\n" + BODY,
            None,
            id="no_title",
        ),
    ],
)
def test_parse_and_split_title(content, expected_title):
    items = parse.parse_and_split(
        content, "user/page.md", "https://gitlab.com/help/user/page"
    )

    assert items
    assert all(item["metadata"]["title"] == expected_title for item in items)
    assert all(not item["content"].startswith("---") for item in items)
