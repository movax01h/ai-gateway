import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.branch import CreateBranch, CreateBranchInput


def create_http_response(data, status_code=200):
    """Helper function to create a GitLabHttpResponse from data."""
    return GitLabHttpResponse(status_code=status_code, body=data)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    mock.apost = AsyncMock()
    return mock


@pytest.fixture(name="project_mock")
def project_mock_fixture():
    """Fixture for mock project."""
    return Project(
        id=24,
        name="test-project",
        description="Test project",
        http_url_to_repo="http://example.com/repo.git",
        web_url="http://example.com/repo",
        languages=[],
        exclusion_rules=[],
    )


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock, project_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": project_mock,
    }


@pytest.fixture(name="branch_data")
def branch_data_fixture():
    """Fixture for sample branch data."""
    return {
        "name": "feature-branch",
        "commit": {
            "id": "6104942438c14ec7bd21c6cd5bd995272b3faff6",
            "short_id": "61049424",
            "title": "Test commit",
            "author_name": "Test User",
            "author_email": "test@example.com",
            "created_at": "2025-04-29T11:35:36.000+02:00",
            "message": "Test commit message",
        },
        "merged": False,
        "protected": False,
        "developers_can_push": False,
        "developers_can_merge": False,
        "can_push": True,
        "default": False,
        "web_url": "http://example.com/repo/-/tree/feature-branch",
    }


@pytest.mark.asyncio
async def test_create_branch_success(gitlab_client_mock, metadata, branch_data):
    """Test successful branch creation with project_id."""
    gitlab_client_mock.apost = AsyncMock(return_value=create_http_response(branch_data))

    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(project_id=24, branch="feature-branch", ref="main")

    expected_response = json.dumps({"status": "success", "branch": branch_data})
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/branches",
        body=json.dumps({"branch": "feature-branch", "ref": "main"}),
    )


@pytest.mark.asyncio
async def test_create_branch_with_url_success(
    gitlab_client_mock, metadata, branch_data
):
    """Test successful branch creation with URL."""
    gitlab_client_mock.apost = AsyncMock(return_value=create_http_response(branch_data))

    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project",
        branch="feature-branch",
        ref="main",
    )

    expected_response = json.dumps({"status": "success", "branch": branch_data})
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/namespace%2Fproject/repository/branches",
        body=json.dumps({"branch": "feature-branch", "ref": "main"}),
    )


@pytest.mark.asyncio
async def test_create_branch_with_commit_sha(gitlab_client_mock, metadata, branch_data):
    """Test branch creation from a commit SHA."""
    gitlab_client_mock.apost = AsyncMock(return_value=create_http_response(branch_data))

    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(
        project_id=24,
        branch="feature-branch",
        ref="6104942438c14ec7bd21c6cd5bd995272b3faff6",
    )

    expected_response = json.dumps({"status": "success", "branch": branch_data})
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/branches",
        body=json.dumps(
            {
                "branch": "feature-branch",
                "ref": "6104942438c14ec7bd21c6cd5bd995272b3faff6",
            }
        ),
    )


@pytest.mark.asyncio
async def test_create_branch_error(gitlab_client_mock, metadata):
    """Test error handling when branch creation fails."""
    error_response = create_http_response(
        {"message": "Branch already exists"}, status_code=400
    )
    gitlab_client_mock.apost = AsyncMock(return_value=error_response)

    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(project_id=24, branch="existing-branch", ref="main")

    response_data = json.loads(response)
    assert "error" in response_data
    assert "HTTP 400" in response_data["error"]

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/branches",
        body=json.dumps({"branch": "existing-branch", "ref": "main"}),
    )


@pytest.mark.asyncio
async def test_create_branch_validation_errors(metadata):
    """Test validation errors for missing required parameters."""
    tool = CreateBranch(description="Create branch", metadata=metadata)

    # Test missing project_id and URL
    response = await tool._arun(branch="feature-branch", ref="main")

    response_data = json.loads(response)
    assert "error" in response_data
    assert "'project_id' must be provided when 'url' is not" in response_data["error"]


@pytest.mark.asyncio
async def test_create_branch_url_mismatch(gitlab_client_mock, metadata):
    """Test error when URL and project_id don't match."""
    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project",
        project_id="different/project",
        branch="feature-branch",
        ref="main",
    )

    response_data = json.loads(response)
    assert "error" in response_data
    assert "Project ID mismatch" in response_data["error"]

    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_create_branch_invalid_url(gitlab_client_mock, metadata):
    """Test error handling for invalid URL."""
    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(
        url="https://example.com/not-gitlab",
        branch="feature-branch",
        ref="main",
    )

    response_data = json.loads(response)
    assert "error" in response_data
    assert "Failed to parse URL" in response_data["error"]

    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateBranchInput(project_id=24, branch="feature-branch", ref="main"),
            "Create branch feature-branch from main in project 24",
        ),
        (
            CreateBranchInput(
                url="https://gitlab.com/namespace/project",
                branch="feature-branch",
                ref="main",
            ),
            "Create branch feature-branch from main in https://gitlab.com/namespace/project",
        ),
        (
            CreateBranchInput(
                project_id=24,
                branch="hotfix-123",
                ref="6104942438c14ec7bd21c6cd5bd995272b3faff6",
            ),
            "Create branch hotfix-123 from 6104942438c14ec7bd21c6cd5bd995272b3faff6 in project 24",
        ),
    ],
)
def test_format_display_message(input_data, expected_message):
    """Test the format_display_message method of CreateBranch."""
    tool = CreateBranch(description="Create branch")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_create_branch_exception(gitlab_client_mock, metadata):
    """Test exception handling in CreateBranch._execute method."""
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))

    tool = CreateBranch(description="Create branch", metadata=metadata)

    response = await tool._arun(project_id=24, branch="feature-branch", ref="main")

    response_data = json.loads(response)
    assert "error" in response_data
    assert error_message in response_data["error"]

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/branches",
        body=json.dumps({"branch": "feature-branch", "ref": "main"}),
    )
