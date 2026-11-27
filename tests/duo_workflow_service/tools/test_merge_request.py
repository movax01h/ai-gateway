import pytest
from pydantic import ValidationError

from ai.features.review.recommend_reviewers import components
from duo_workflow_service.tools import merge_request
from duo_workflow_service.tools.merge_request import (
    CreateMergeRequestInput,
    UpdateMergeRequestInput,
)

USER_ID_INPUTS = [
    pytest.param(
        CreateMergeRequestInput,
        {"source_branch": "feature", "target_branch": "main", "title": "Title"},
        id="create",
    ),
    pytest.param(UpdateMergeRequestInput, {"merge_request_iid": 1}, id="update"),
]


@pytest.mark.parametrize("input_cls,required_args", USER_ID_INPUTS)
@pytest.mark.parametrize("field", ["assignee_ids", "reviewer_ids"])
@pytest.mark.parametrize(
    "value,expected",
    [
        ("[1]", [1]),
        ("[1, 2]", [1, 2]),
        ("[]", []),
        ([1, 2], [1, 2]),
        (None, None),
    ],
)
def test_user_ids_accept_json_encoded_list(
    input_cls, required_args, field, value, expected
):
    input_data = input_cls(**required_args, **{field: value})

    assert getattr(input_data, field) == expected


@pytest.mark.parametrize("input_cls,required_args", USER_ID_INPUTS)
@pytest.mark.parametrize("field", ["assignee_ids", "reviewer_ids"])
@pytest.mark.parametrize(
    "value,expected_error",
    [
        ("not valid json", "Invalid JSON string"),
        ("1", "Expected JSON string to deserialize to a list, got int"),
        ('["a"]', "Input should be a valid integer"),
    ],
)
def test_user_ids_reject_invalid_json_string(
    input_cls, required_args, field, value, expected_error
):
    with pytest.raises(ValidationError, match=expected_error):
        input_cls(**required_args, **{field: value})


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
        match=rf"merge_request\.{name} moved to ai\.features\.review\.recommend_reviewers\.components",
    ):
        moved = getattr(merge_request, name)

    assert moved is getattr(components, name)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'NotATool'"):
        getattr(merge_request, "NotATool")


def test_dir_lists_the_moved_tools():
    assert set(merge_request._MOVED_TOOLS) <= set(dir(merge_request))
