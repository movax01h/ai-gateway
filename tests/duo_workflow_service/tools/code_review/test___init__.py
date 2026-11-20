import pytest

from ai.features.review.code_review import components
from duo_workflow_service.tools import code_review


@pytest.mark.parametrize(
    "name",
    [
        "DetectReviewContextTruncation",
        "FinalizeCodeReviewFindings",
        "PostDuoCodeReview",
        "PostDuoCodeReviewFindings",
    ],
)
def test_moved_tool_resolves_to_the_feature_class(name):
    with pytest.warns(
        DeprecationWarning,
        match=rf"code_review\.{name} moved to ai\.features\.review\.code_review\.components",
    ):
        moved = getattr(code_review, name)

    assert moved is getattr(components, name)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'NotATool'"):
        getattr(code_review, "NotATool")


def test_dir_lists_the_moved_tools():
    assert {
        "BuildReviewMergeRequestContext",
        "DetectReviewContextTruncation",
        "FinalizeCodeReviewFindings",
        "PostDuoCodeReview",
        "PostDuoCodeReviewFindings",
    } <= set(dir(code_review))
