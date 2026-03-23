import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    CreateMergeRequest,
    CreateMergeRequestInput,
    MergeRequestResourceInput,
)


@pytest.fixture(name="merge_request_data")
def merge_request_data_fixture():
    """Fixture for common merge request data."""
    return {
        "id": 1,
        "title": "Test Merge Request",
        "source_branch": "feature",
        "target_branch": "main",
    }


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
    response = await tool._arun(
        url=url, project_id=project_id, merge_request_iid=merge_request_iid, **kwargs
    )

    error_response = json.loads(response)
    assert "error" in error_response
    assert error_contains in error_response["error"]

    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.apost.assert_not_called()
    gitlab_client_mock.aput.assert_not_called()


@pytest.mark.asyncio
async def test_create_merge_request(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={
            "id": 1,
            "title": "New Feature",
            "source_branch": "feature",
            "target_branch": "main",
        },
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequest(metadata=metadata)

    input_data = {
        "project_id": 1,
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
        "description": "Feature description",
        "assignee_ids": [123],
        "reviewer_ids": [456],
        "remove_source_branch": True,
        "squash": True,
        "labels": "bug,feature",
    }

    response = await tool.arun(input_data)

    expected_data = {
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
        "description": "Feature description",
        "assignee_ids": [123],
        "reviewer_ids": [456],
        "remove_source_branch": True,
        "squash": True,
        "labels": "bug,feature",
    }

    expected_response = json.dumps(
        {
            "created_merge_request": {
                "id": 1,
                "title": "New Feature",
                "source_branch": "feature",
                "target_branch": "main",
            },
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests",
        body=json.dumps(expected_data),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    [
        # Test with only URL for project
        (
            "https://gitlab.com/namespace/project",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests",
        ),
        # Test with URL and matching project_id
        (
            "https://gitlab.com/namespace/project",
            "namespace%2Fproject",
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests",
        ),
    ],
)
async def test_create_merge_request_with_url_success(
    url,
    project_id,
    merge_request_iid,
    expected_path,
    gitlab_client_mock,
    metadata,
    merge_request_data,
):
    tool = CreateMergeRequest(
        description="create merge request description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=merge_request_data,
        source_branch="feature",
        target_branch="main",
        title="Test Merge Request",
    )

    expected_response = json.dumps(
        {
            "created_merge_request": {
                "id": 1,
                "title": "Test Merge Request",
                "source_branch": "feature",
                "target_branch": "main",
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_path,
        body=json.dumps(
            {
                "source_branch": "feature",
                "target_branch": "main",
                "title": "Test Merge Request",
            }
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    [
        # URL and project_id both given, but don't match
        (
            "https://gitlab.com/namespace/project",
            "different%2Fproject",
            None,
            "Project ID mismatch",
        ),
        # URL given isn't a valid GitLab URL
        (
            "https://example.com/not-gitlab",
            None,
            None,
            "Failed to parse URL",
        ),
    ],
)
async def test_create_merge_request_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = CreateMergeRequest(
        description="create merge request description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        source_branch="feature",
        target_branch="main",
        title="Test Merge Request",
    )


@pytest.mark.asyncio
async def test_create_merge_request_with_labels(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={
            "id": 1,
            "title": "Bug Fix",
            "source_branch": "bugfix",
            "target_branch": "main",
            "labels": ["bug", "urgent"],
        },
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequest(metadata=metadata)

    input_data = {
        "project_id": 1,
        "source_branch": "bugfix",
        "target_branch": "main",
        "title": "Bug Fix",
        "labels": "bug,urgent",
    }

    response = await tool.arun(input_data)

    expected_data = {
        "source_branch": "bugfix",
        "target_branch": "main",
        "title": "Bug Fix",
        "labels": "bug,urgent",
    }

    expected_response = json.dumps(
        {
            "created_merge_request": {
                "id": 1,
                "title": "Bug Fix",
                "source_branch": "bugfix",
                "target_branch": "main",
                "labels": ["bug", "urgent"],
            },
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests",
        body=json.dumps(expected_data),
    )


@pytest.mark.asyncio
async def test_create_merge_request_minimal_params(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={
            "id": 1,
            "title": "New Feature",
            "source_branch": "feature",
            "target_branch": "main",
        },
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequest(metadata=metadata)

    input_data = {
        "project_id": 1,
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
    }

    response = await tool.arun(input_data)

    expected_data = {
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
    }

    expected_response = json.dumps(
        {
            "created_merge_request": {
                "id": 1,
                "title": "New Feature",
                "source_branch": "feature",
                "target_branch": "main",
            },
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests",
        body=json.dumps(expected_data),
    )


@pytest.mark.asyncio
async def test_create_merge_request_with_server_error(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=409,
        body={"status": 409, "message": "Duplicate request"},
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequest(metadata=metadata)

    input_data = {
        "project_id": 1,
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
    }

    response = await tool.arun(input_data)

    expected_data = {
        "source_branch": "feature",
        "target_branch": "main",
        "title": "New Feature",
    }

    response_json = json.loads(response)
    assert "HTTP 409" in response_json["error"]
    assert "{'status': 409, 'message': 'Duplicate request'}" in response_json["error"]

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests",
        body=json.dumps(expected_data),
    )


@pytest.mark.asyncio
async def test_create_merge_request_exception(gitlab_client_mock, metadata):
    """Test exception handling in CreateMergeRequest._arun method."""
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))

    tool = CreateMergeRequest(metadata=metadata)

    response = await tool._arun(
        project_id=1, source_branch="feature", target_branch="main", title="Test MR"
    )

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateMergeRequestInput(
                project_id=42,
                source_branch="feature-branch",
                target_branch="main",
                title="New feature implementation",
                description="This implements the new feature",
                assignee_ids=[123, 456],
                reviewer_ids=[789],
                remove_source_branch=True,
                squash=True,
            ),
            "Create merge request from 'feature-branch' to 'main' in project 42",
        ),
        (
            CreateMergeRequestInput(
                url="https://gitlab.com/namespace/project",
                source_branch="feature-branch",
                target_branch="main",
                title="New feature implementation",
                description="This implements the new feature",
                assignee_ids=[123, 456],
                reviewer_ids=[789],
                remove_source_branch=True,
                squash=True,
            ),
            "Create merge request from 'feature-branch' to 'main' in https://gitlab.com/namespace/project",
        ),
    ],
)
def test_create_merge_request_format_display_message(input_data, expected_message):
    tool = CreateMergeRequest(description="Create merge request")
    message = tool.format_display_message(input_data)
    assert message == expected_message
