import pytest

from ai_gateway.model_selection.keyword_bm25 import KeywordBM25, words


def test_words_share_one_form_across_endings():
    assert words("Rename renamed renaming typos") == ["renam", "renam", "renam", "typo"]


def test_words_ignore_links_and_mentions():
    assert words("@user see https://gitlab.com/acme/readme-gen/-/issues/7") == ["see"]


@pytest.mark.parametrize(
    ("goal", "matched"),
    [
        ("Fix a typo in the README", ["typo", "readme"]),
        ("Fix the race condition in the scheduler", ["race condition"]),
        ("Fix the race in the scheduler, the race happens on shutdown", []),
        ("Add pagination to the issues API", []),
    ],
)
def test_matches(goal, matched):
    keywords = ["refactor", "race condition", "typo", "readme", "rename"]

    hits = KeywordBM25(keywords).matches(goal)

    assert [k for k, hit in zip(keywords, hits) if hit] == matched


def test_rejects_an_empty_keyword_list():
    with pytest.raises(ValueError, match="at least one keyword"):
        KeywordBM25([])
