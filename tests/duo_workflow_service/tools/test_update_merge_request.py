import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools.base import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    UpdateMergeRequest,
    UpdateMergeRequestInput,
)

# Common URL test parameters
URL_SUCCESS_CASES = [
    # Test with only URL
    (
        "https://gitlab.com/namespace/project/-/merge_requests/123",
        None,
        None,
        "/api/v4/projects/namespace%2Fproject/merge_requests/123",
    ),
    # Test with URL and matching project_id and merge_request_iid
    (
        "https://gitlab.com/namespace/project/-/merge_requests/123",
        "namespace%2Fproject",
        123,
        "/api/v4/projects/namespace%2Fproject/merge_requests/123",
    ),
]

URL_ERROR_CASES = [
    # URL and project_id both given, but don't match
    (
        "https://gitlab.com/namespace/project/-/merge_requests/123",
        "different%2Fproject",
        123,
        "Project ID mismatch",
    ),
    # URL and merge_request_iid both given, but don't match
    (
        "https://gitlab.com/namespace/project/-/merge_requests/123",
        "namespace%2Fproject",
        456,
        "Merge Request ID mismatch",
    ),
    # URL given isn't a merge request URL (it's just a project URL)
    (
        "https://gitlab.com/namespace/project",
        None,
        None,
        "Failed to parse URL",
    ),
]


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="project_mock")
def project_mock_fixture():
    """Fixture for mock project with exclusion rules."""
    return Project(
        id=1,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=["**/*.log", "/secrets/**", "**/node_modules/**"],
    )


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock, project_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project_mock,
    }


async def tool_url_success_response(
    tool,
    url,
    project_id,
    merge_request_iid,
    gitlab_client_mock,
    response_data,
    **kwargs,
):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=response_data,
        headers={"content-type": "application/json"},
    )

    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)
    gitlab_client_mock.aput = AsyncMock(return_value=mock_response)

    response = await tool._arun(
        url=url, project_id=project_id, merge_request_iid=merge_request_iid, **kwargs
    )

    return response


async def assert_tool_url_error(
    tool,
    url,
    project_id,
    merge_request_iid,
    error_contains,
    gitlab_client_mock,
    **kwargs,
):
    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            url=url,
            project_id=project_id,
            merge_request_iid=merge_request_iid,
            **kwargs,
        )

    assert error_contains in str(exc_info.value)

    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.apost.assert_not_called()
    gitlab_client_mock.aput.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    URL_SUCCESS_CASES,
)
async def test_update_merge_request_with_url_success(
    url, project_id, merge_request_iid, expected_path, gitlab_client_mock, metadata
):
    update_data = {
        "id": 123,
        "title": "Updated Test MR",
        "description": "This is an updated test merge request",
    }

    mock_response = GitLabHttpResponse(
        status_code=200, body=update_data, headers={"content-type": "application/json"}
    )
    gitlab_client_mock.aput = AsyncMock(return_value=mock_response)

    tool = UpdateMergeRequest(
        description="update merge request description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=update_data,
        title="Updated Test MR",
        description="This is an updated test merge request",
    )

    expected_response = json.dumps(
        {
            "updated_merge_request": {
                "id": 123,
                "title": "Updated Test MR",
                "description": "This is an updated test merge request",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path=expected_path,
        body=json.dumps(
            {
                "title": "Updated Test MR",
                "description": "This is an updated test merge request",
            }
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_update_merge_request_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = UpdateMergeRequest(
        description="update merge request description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        title="Updated Test MR",
    )


@pytest.mark.asyncio
async def test_update_merge_request(gitlab_client_mock, metadata):
    update_data = {"id": 123, "title": "Updated MR", "description": "New description"}
    mock_response = GitLabHttpResponse(
        status_code=200, body=update_data, headers={"content-type": "application/json"}
    )
    gitlab_client_mock.aput = AsyncMock(return_value=mock_response)
    tool = UpdateMergeRequest(metadata=metadata)

    response = await tool.arun(
        {
            "project_id": 1,
            "merge_request_iid": 123,
            "title": "Updated MR",
            "description": "New description",
            "remove_source_branch": True,
        }
    )

    expected_response = json.dumps({"updated_merge_request": update_data})
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123",
        body=json.dumps(
            {
                "description": "New description",
                "remove_source_branch": True,
                "title": "Updated MR",
            }
        ),
    )


@pytest.mark.asyncio
async def test_update_merge_request_with_labels(gitlab_client_mock, metadata):
    update_data = {
        "id": 123,
        "title": "Bug Fix",
        "labels": ["bug", "urgent", "high-priority"],
    }
    mock_response = GitLabHttpResponse(
        status_code=200, body=update_data, headers={"content-type": "application/json"}
    )
    gitlab_client_mock.aput = AsyncMock(return_value=mock_response)
    tool = UpdateMergeRequest(metadata=metadata)

    response = await tool.arun(
        {
            "project_id": 1,
            "merge_request_iid": 123,
            "title": "Bug Fix",
            "labels": "bug,urgent,high-priority",
        }
    )

    expected_response = json.dumps({"updated_merge_request": update_data})
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123",
        body=json.dumps(
            {
                "title": "Bug Fix",
                "labels": "bug,urgent,high-priority",
            }
        ),
    )


@pytest.mark.asyncio
async def test_update_merge_request_http_error_status_code(
    gitlab_client_mock, metadata
):
    """Test that HTTP error status codes raise ToolException rather than returning error JSON."""
    gitlab_client_mock.aput = AsyncMock(
        return_value=GitLabHttpResponse(
            status_code=404,
            body={"message": "404 Merge Request Not Found"},
            headers={"content-type": "application/json"},
        )
    )

    tool = UpdateMergeRequest(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_id=1, merge_request_iid=999, title="Updated MR")

    assert "HTTP 404" in str(exc_info.value)
    assert "404 Merge Request Not Found" in str(exc_info.value)

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/999",
        body=json.dumps({"title": "Updated MR"}),
    )


@pytest.mark.asyncio
async def test_update_merge_request_exception(gitlab_client_mock, metadata):
    """Test that exceptions from UpdateMergeRequest._execute propagate rather than being swallowed."""
    error_message = "API error"
    gitlab_client_mock.aput = AsyncMock(side_effect=Exception(error_message))

    tool = UpdateMergeRequest(metadata=metadata)

    with pytest.raises(Exception, match=error_message):
        await tool._arun(project_id=1, merge_request_iid=123, title="Updated MR")


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            UpdateMergeRequestInput(
                project_id=42,
                merge_request_iid=123,
                title="Updated feature implementation",
                description="Updated description",
                allow_collaboration=True,
                assignee_ids=[123, 456],
                discussion_locked=True,
                milestone_id=10,
                remove_source_branch=True,
                reviewer_ids=[789],
                squash=True,
                state_event="close",
                target_branch="develop",
            ),
            "Update merge request !123 in project 42",
        ),
        (
            UpdateMergeRequestInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                title="Updated title",
                description="Updated description",
            ),
            "Update merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_update_merge_request_format_display_message(input_data, expected_message):
    tool = UpdateMergeRequest(description="Update merge request")
    message = tool.format_display_message(input_data)
    assert message == expected_message
