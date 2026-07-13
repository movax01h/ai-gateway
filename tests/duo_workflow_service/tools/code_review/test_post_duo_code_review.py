import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.code_review.post_duo_code_review import (
    PostDuoCodeReview,
    PostDuoCodeReviewInput,
)


@pytest.mark.asyncio
async def test_post_duo_code_review(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=200, body=json.dumps({"message": "Comments added successfully"})
        )
    )
    tool = PostDuoCodeReview(metadata=metadata)
    response = await tool._arun(
        project_id="123", merge_request_iid=45, review_output="<review>test</review>"
    )
    expected_response = json.dumps(
        {"status": "success", "message": "Review posted to MR !45"}
    )
    assert response == expected_response
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/code_review/add_comments",
        body=json.dumps(
            {
                "project_id": "123",
                "merge_request_iid": 45,
                "review_output": "<review>test</review>",
                "workflow_id": "test-workflow-123",
            }
        ),
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_post_duo_code_review_exception(gitlab_client_mock, metadata):
    """Test that exceptions from PostDuoCodeReview._execute propagate rather than being swallowed."""
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))
    tool = PostDuoCodeReview(metadata=metadata)
    with pytest.raises(Exception, match=error_message):
        await tool._arun(
            project_id=123, merge_request_iid=45, review_output="<review>test</review>"
        )


@pytest.mark.asyncio
async def test_post_duo_code_review_failure_response(gitlab_client_mock, metadata):
    """Test that a non-success API response raises ToolException."""
    gitlab_client_mock.apost = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=422, body=json.dumps({"message": "Validation failed"})
        )
    )
    tool = PostDuoCodeReview(metadata=metadata)
    with pytest.raises(ToolException, match="Failed to post review"):
        await tool._arun(
            project_id=123, merge_request_iid=45, review_output="<review>test</review>"
        )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            PostDuoCodeReviewInput(
                project_id=42,
                merge_request_iid=123,
                review_output="<review>test</review>",
            ),
            "Post Duo Code Review to merge request !123 in project 42",
        ),
    ],
)
def test_post_duo_code_review_format_display_message(input_data, expected_message):
    tool = PostDuoCodeReview(description="Post Duo Code Review")
    message = tool.format_display_message(input_data)
    assert message == expected_message
