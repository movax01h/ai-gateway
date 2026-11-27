import pytest

from ai.features.review.recommend_reviewers import components
from duo_workflow_service import tools


@pytest.mark.parametrize(
    "name",
    [
        "AddMergeRequestReviewers",
        "AddMergeRequestReviewersInput",
        "PostSuggestedReviewers",
        "PostSuggestedReviewersInput",
        "SuggestedReviewerInput",
    ],
)
def test_moved_tool_resolves_to_the_feature_class(name):
    with pytest.warns(
        DeprecationWarning,
        match=rf"tools\.{name} moved to ai\.features\.review\.recommend_reviewers\.components",
    ):
        moved = getattr(tools, name)

    assert moved is getattr(components, name)
