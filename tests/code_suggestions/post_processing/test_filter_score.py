import pytest

from ai_gateway.code_suggestions.processing.post.ops import filter_score


def test_filter_score_below_threshold():
    # Test when score is below threshold
    assert filter_score("some code", 0.3, 0.5) == ""
    assert filter_score("print('hello')", 0.1, 0.2) == ""


def test_filter_score_above_threshold():
    # Test when score is above threshold
    assert filter_score("some code", 0.7, 0.5) == "some code"
    assert filter_score("print('hello')", 0.9, 0.8) == "print('hello')"


def test_filter_score_equal_threshold():
    # Test when score equals threshold
    assert filter_score("some code", 0.5, 0.5) == "some code"


def test_filter_score_integer_values():
    # Test with integer scores
    assert filter_score("some code", 0, 1) == ""
    assert filter_score("some code", 2, 1) == "some code"


def test_filter_score_invalid_types():
    # Test with invalid score types
    assert filter_score("some code", "invalid", 0.5) == "some code"
    assert filter_score("some code", None, 0.5) == "some code"
    assert filter_score("some code", [], 0.5) == "some code"


def test_filter_score_edge_cases():
    # Test edge cases
    assert filter_score("", 0.3, 0.5) == ""  # Empty completion with low score
    assert filter_score("", 0.7, 0.5) == ""  # Empty completion with high score
    assert filter_score("some code", float("inf"), 0.5) == "some code"
    assert filter_score("some code", float("-inf"), 0.5) == ""


@pytest.mark.parametrize(
    "completion,score,threshold,expected",
    [
        ("code", 0.3, 0.5, ""),
        ("code", 0.7, 0.5, "code"),
        ("code", 0.5, 0.5, "code"),
        ("", 0.3, 0.5, ""),
        ("code", "invalid", 0.5, "code"),
        ("code", None, 0.5, "code"),
    ],
)
def test_filter_score_parametrized(completion, score, threshold, expected):
    assert filter_score(completion, score, threshold) == expected
