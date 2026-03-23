import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    ListMergeRequest,
    ListMergeRequestInput,
)


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


@pytest.mark.asyncio
async def test_list_merge_request_basic(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=[{"id": 1, "title": "Test MR", "author": {"username": "testuser"}}],
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListMergeRequest(metadata=metadata)

    response = await tool._arun(project_id=1)

    expected_response = json.dumps(
        {
            "merge_requests": [
                {"id": 1, "title": "Test MR", "author": {"username": "testuser"}}
            ]
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests",
        params={},
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_list_merge_request_with_filters(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=[{"id": 1, "title": "Test MR", "author": {"username": "testuser"}}],
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListMergeRequest(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        author_username="testuser",
        state="opened",
        labels="bug,urgent",
        search="test search",
    )

    expected_response = json.dumps(
        {
            "merge_requests": [
                {"id": 1, "title": "Test MR", "author": {"username": "testuser"}}
            ]
        }
    )
    assert response == expected_response

    expected_params = {
        "author_username": "testuser",
        "state": "opened",
        "labels": "bug,urgent",
        "search": "test search",
    }
    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests",
        params=expected_params,
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,expected_path",
    [
        (
            "https://gitlab.com/namespace/project",
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests",
        ),
        (
            "https://gitlab.com/namespace/project",
            "namespace%2Fproject",
            "/api/v4/projects/namespace%2Fproject/merge_requests",
        ),
    ],
)
async def test_list_merge_request_with_url_success(
    url, project_id, expected_path, gitlab_client_mock, metadata
):
    mock_response_data = [{"id": 1, "title": "Test MR"}]
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=mock_response_data,
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListMergeRequest(metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id)

    expected_response = json.dumps({"merge_requests": mock_response_data})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, params={}, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,error_contains",
    [
        (
            "https://gitlab.com/namespace/project",
            "different%2Fproject",
            "Project ID mismatch",
        ),
        (
            "https://example.com/not-gitlab",
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_list_merge_request_with_url_error(
    url, project_id, error_contains, gitlab_client_mock, metadata
):
    tool = ListMergeRequest(metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id)

    error_response = json.loads(response)
    assert "error" in error_response
    assert error_contains in error_response["error"]

    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_list_merge_request_exception(gitlab_client_mock, metadata):
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = ListMergeRequest(metadata=metadata)

    response = await tool._arun(project_id=1)

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListMergeRequestInput(project_id=42),
            "List merge requests in project 42 ",
        ),
        (
            ListMergeRequestInput(
                project_id=42,
                author_username="testuser",
                state="opened",
                labels="bug,urgent",
            ),
            "List merge requests in project 42 with filters: author: testuser, state: opened, labels: bug,urgent",
        ),
        (
            ListMergeRequestInput(
                url="https://gitlab.com/namespace/project", author_username="testuser"
            ),
            "List merge requests in https://gitlab.com/namespace/project with filters: author: testuser",
        ),
    ],
)
def test_list_merge_request_format_display_message(input_data, expected_message):
    tool = ListMergeRequest(description="List merge requests")
    message = tool.format_display_message(input_data)
    assert message == expected_message
