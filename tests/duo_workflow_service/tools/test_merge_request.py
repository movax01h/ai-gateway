import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    PostSuggestedReviewers,
    PostSuggestedReviewersInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture(name="success_response_mock")
def success_response_mock_fixture():
    return GitLabHttpResponse(
        status_code=201,
        body={
            "suggestions": [
                {"user_id": 42, "reason": "Owns the affected module"},
                {"user_id": 57, "reason": "Recently reviewed related code"},
            ]
        },
    )


@pytest.mark.asyncio
async def test_post_suggested_reviewers_success(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSuggestedReviewers(metadata=metadata)

    input_data = {
        "project_id": 13,
        "merge_request_iid": 9,
        "suggestions": [
            {
                "user_id": 42,
                "reason": "Owns the affected module",
                "approval_rule_id": 7,
                "approval_rule_type": "merge_request_rule",
            },
            {"user_id": 57},
        ],
    }

    response = await tool.arun(input_data)

    response_data = json.loads(response)
    assert response_data["suggested_reviewers"] == success_response_mock.body

    gitlab_client_mock.apost.assert_called_once()
    call_args = gitlab_client_mock.apost.call_args
    assert (
        call_args[1]["path"]
        == "/api/v4/projects/13/merge_requests/9/suggested_reviewers"
    )

    body = json.loads(call_args[1]["body"])
    assert body["suggestions"] == [
        {
            "user_id": 42,
            "reason": "Owns the affected module",
            "approval_rule_id": 7,
            "approval_rule_type": "merge_request_rule",
        },
        {"user_id": 57},
    ]


@pytest.mark.asyncio
async def test_post_suggested_reviewers_empty_suggestions(
    gitlab_client_mock, metadata, success_response_mock
):
    gitlab_client_mock.apost = AsyncMock(return_value=success_response_mock)

    tool = PostSuggestedReviewers(metadata=metadata)

    input_data = {
        "project_id": 13,
        "merge_request_iid": 9,
        "suggestions": [],
    }

    await tool.arun(input_data)

    call_args = gitlab_client_mock.apost.call_args
    body = json.loads(call_args[1]["body"])
    assert body["suggestions"] == []


@pytest.mark.asyncio
async def test_post_suggested_reviewers_api_error_with_status(
    gitlab_client_mock, metadata
):
    error_response = GitLabHttpResponse(
        status_code=404,
        body={"message": "Merge request not found"},
    )

    gitlab_client_mock.apost = AsyncMock(return_value=error_response)

    tool = PostSuggestedReviewers(metadata=metadata)

    input_data = {
        "project_id": 13,
        "merge_request_iid": 999,
        "suggestions": [{"user_id": 42}],
    }

    with pytest.raises(ToolException) as exc_info:
        await tool.arun(input_data)

    assert "HTTP 404" in str(exc_info.value)


@pytest.mark.asyncio
async def test_post_suggested_reviewers_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network Error"))

    tool = PostSuggestedReviewers(metadata=metadata)

    input_data = {
        "project_id": 13,
        "merge_request_iid": 9,
        "suggestions": [{"user_id": 42}],
    }

    with pytest.raises(Exception, match="Network Error"):
        await tool.arun(input_data)


def test_post_suggested_reviewers_format_display_message():
    tool = PostSuggestedReviewers(metadata={})
    input_data = PostSuggestedReviewersInput(
        project_id=13,
        merge_request_iid=9,
        suggestions=[
            {"user_id": 42, "reason": "Owns the affected module"},
            {"user_id": 57},
        ],
    )
    expected_message = "Save 2 suggested reviewer(s) for merge request !9 in project 13"
    assert tool.format_display_message(input_data) == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["19.3.0", "18.6.0"])
async def test_post_suggested_reviewers_older_instance_does_not_call_gitlab(
    gitlab_client_mock, metadata, version
):
    # 19.4 is when the endpoint landed. Below it the POST can only 404, so it is
    # never issued.
    gitlab_client_mock.apost = AsyncMock()
    tool = PostSuggestedReviewers(metadata=metadata)

    with patch(
        "duo_workflow_service.tools.version_compatibility.gitlab_version"
    ) as mock_version:
        mock_version.get.return_value = version

        with pytest.raises(ToolException) as exc_info:
            await tool.arun(
                {
                    "project_id": 13,
                    "merge_request_iid": 9,
                    "suggestions": [{"user_id": 42}],
                }
            )

    assert "requires GitLab 19.4 or later" in str(exc_info.value)
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["19.4.0-pre", "19.5.0", None])
async def test_post_suggested_reviewers_supported_or_unknown_version_proceeds(
    gitlab_client_mock, metadata, version
):
    # 19.4.0-pre is what GitLab.com reports, and an unreported version must not
    # block the call: the version_compatibility fallback predates 19.4.
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=200, body={"suggestions": []})
    )
    tool = PostSuggestedReviewers(metadata=metadata)

    with patch(
        "duo_workflow_service.tools.version_compatibility.gitlab_version"
    ) as mock_version:
        mock_version.get.return_value = version

        await tool.arun(
            {"project_id": 13, "merge_request_iid": 9, "suggestions": [{"user_id": 42}]}
        )

    gitlab_client_mock.apost.assert_called_once()
