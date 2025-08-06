import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.commit import (
    CommitResourceInput,
    CreateCommit,
    CreateCommitAction,
    CreateCommitInput,
    GetCommit,
    GetCommitComments,
    GetCommitDiff,
    ListCommits,
    ListCommitsInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    return mock


@pytest.fixture(name="project_mock")
def project_mock_fixture():
    """Fixture for mock project with exclusion rules."""
    return Project(
        id=24,
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


@pytest.fixture(name="commit_data")
def commit_data_fixture():
    """Fixture for sample commit data."""
    return {
        "id": "6104942438c14ec7bd21c6cd5bd995272b3faff6",
        "short_id": "61049424",
        "title": "Test commit",
        "author_name": "Test User",
        "author_email": "test@example.com",
        "created_at": "2025-04-29T11:35:36.000+02:00",
        "message": "Test commit message",
    }


@pytest.mark.asyncio
async def test_get_commit(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value={
            "id": "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "short_id": "c34bb66f",
            "created_at": "2025-04-29T11:35:36.000+02:00",
            "title": "Test Commit",
            "message": "Test Commit message",
            "author_name": "Author Name",
            "author_email": "test@example.com",
            "authored_date": "2025-04-29T11:35:36.000+02:00",
            "committed_date": "2025-04-29T11:35:36.000+02:00",
            "stats": {"additions": 591, "deletions": 4, "total": 595},
        }
    )

    tool = GetCommit(description="get commit description", metadata=metadata)

    response = await tool._arun(
        project_id=24, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    )

    expected_response = json.dumps(
        {
            "commit": {
                "id": "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
                "short_id": "c34bb66f",
                "created_at": "2025-04-29T11:35:36.000+02:00",
                "title": "Test Commit",
                "message": "Test Commit message",
                "author_name": "Author Name",
                "author_email": "test@example.com",
                "authored_date": "2025-04-29T11:35:36.000+02:00",
                "committed_date": "2025-04-29T11:35:36.000+02:00",
                "stats": {"additions": 591, "deletions": 4, "total": 595},
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        params={},
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_get_commit_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Commit not found"))

    tool = GetCommit(description="get commit description", metadata=metadata)

    response = await tool._arun(project_id=24, commit_sha="nonexistent")

    expected_response = json.dumps({"error": "Commit not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/nonexistent",
        params={},
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_validate_commit_url_no_url_no_ids(metadata):
    tool = GetCommit(description="get commit description", metadata=metadata)

    validation_result = tool._validate_commit_url(
        url=None, project_id=None, commit_sha=None
    )

    assert validation_result.project_id is None
    assert validation_result.commit_sha is None
    assert len(validation_result.errors) == 2
    assert (
        "'project_id' must be provided when 'url' is absent" in validation_result.errors
    )
    assert (
        "'commit_sha' must be provided when 'url' is absent" in validation_result.errors
    )


@pytest.mark.asyncio
async def test_validate_commit_url_no_url_no_project_id(metadata):
    tool = GetCommit(description="get commit description", metadata=metadata)

    validation_result = tool._validate_commit_url(
        url=None, project_id=None, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    )

    assert validation_result.project_id is None
    assert validation_result.commit_sha == "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    assert len(validation_result.errors) == 1
    assert (
        "'project_id' must be provided when 'url' is absent" in validation_result.errors
    )


@pytest.mark.asyncio
async def test_validate_commit_url_no_url_no_commit_sha(metadata):
    tool = GetCommit(description="get commit description", metadata=metadata)

    validation_result = tool._validate_commit_url(
        url=None, project_id="namespace/project", commit_sha=None
    )

    assert validation_result.project_id == "namespace/project"
    assert validation_result.commit_sha is None
    assert len(validation_result.errors) == 1
    assert (
        "'commit_sha' must be provided when 'url' is absent" in validation_result.errors
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,commit_sha,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        ),
        # Test with URL and matching project_id and commit_sha
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "namespace%2Fproject",  # Změněno z "namespace/project" na URL-encoded formát
            "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "/api/v4/projects/namespace%2Fproject/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        ),
    ],
)
async def test_get_commit_with_url_success(
    url, project_id, commit_sha, expected_path, gitlab_client_mock, metadata
):
    mock_response = {
        "id": "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        "short_id": "c34bb66f",
        "created_at": "2025-04-29T11:35:36.000+02:00",
        "title": "Test Commit",
        "message": "Teste Commit message",
        "author_name": "Author Name",
        "author_email": "test@example.com",
        "authored_date": "2025-04-29T11:35:36.000+02:00",
        "committed_date": "2025-04-29T11:35:36.000+02:00",
    }
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetCommit(description="get commit description", metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id, commit_sha=commit_sha)

    expected_response = json.dumps({"commit": mock_response})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, params={}, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,commit_sha,error_contains",
    [
        # URL and project_id both given, but don't match
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "different/project",
            "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "Project ID mismatch",
        ),
        # URL and commit_sha both given, but don't match
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "namespace/project",
            "different_sha",
            "Commit SHA mismatch",
        ),
        # URL given isn't a commit URL (it's just a project URL)
        ("https://gitlab.com/namespace/project", None, None, "Failed to parse URL"),
    ],
)
async def test_get_commit_with_url_error(
    url, project_id, commit_sha, error_contains, gitlab_client_mock, metadata
):
    tool = GetCommit(description="get commit description", metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id, commit_sha=commit_sha)
    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CommitResourceInput(
                project_id=123, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
            ),
            "Read commit c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6 in project 123",
        ),
        (
            CommitResourceInput(
                url="https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
            ),
            "Read commit https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        ),
    ],
)
def test_get_commit_format_display_message(input_data, expected_message):
    tool = GetCommit(description="Get commit description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.parametrize(
    "test_params,expected_params,expected_project_id,description",
    [
        # Basic parameters
        (
            {"project_id": 123, "all": True, "author": "john@example.com"},
            {"all": True, "author": "john@example.com"},
            123,
            "Basic parameters with all=True and author",
        ),
        # Date filtering
        (
            {
                "project_id": 123,
                "since": "2024-01-01T00:00:00Z",
                "until": "2024-01-31T23:59:59Z",
            },
            {"since": "2024-01-01T00:00:00Z", "until": "2024-01-31T23:59:59Z"},
            123,
            "Date filtering with since and until",
        ),
        # Path and branch filtering
        (
            {
                "project_id": 123,
                "path": "src/main.py",
                "ref_name": "develop",
                "first_parent": True,
            },
            {"path": "src/main.py", "ref_name": "develop", "first_parent": True},
            123,
            "Path and branch filtering with first_parent",
        ),
        # Stats and trailers
        (
            {"project_id": 123, "with_stats": True, "trailers": True, "order": "topo"},
            {"with_stats": True, "trailers": True, "order": "topo"},
            123,
            "Stats, trailers, and topological order",
        ),
        # Pagination
        ({"project_id": 123, "page": 2}, {"page": 2}, 123, "Pagination with page 2"),
        # All parameters combined
        (
            {
                "project_id": 123,
                "all": False,
                "author": "jane@example.com",
                "first_parent": False,
                "order": "default",
                "path": "tests/",
                "ref_name": "main",
                "since": "2024-01-01T00:00:00Z",
                "until": "2024-12-31T23:59:59Z",
                "trailers": True,
                "with_stats": True,
                "page": 1,
            },
            {
                "all": False,
                "author": "jane@example.com",
                "first_parent": False,
                "order": "default",
                "path": "tests/",
                "ref_name": "main",
                "since": "2024-01-01T00:00:00Z",
                "until": "2024-12-31T23:59:59Z",
                "trailers": True,
                "with_stats": True,
                "page": 1,
            },
            123,
            "All parameters combined",
        ),
        # URL instead of project_id
        (
            {
                "url": "https://gitlab.com/namespace/project",
                "author": "test@example.com",
            },
            {"author": "test@example.com"},
            "namespace%2Fproject",
            "URL parameter with author",
        ),
        # None values should be filtered out
        (
            {
                "project_id": 123,
                "all": True,
                "author": None,
                "path": "src/",
                "ref_name": None,
                "since": "2024-01-01T00:00:00Z",
                "until": None,
                "page": None,
            },
            {"all": True, "path": "src/", "since": "2024-01-01T00:00:00Z"},
            123,
            "None values filtered out",
        ),
        # Empty parameters (only project_id)
        ({"project_id": 123}, {}, 123, "Empty parameters only project_id"),
    ],
)
@pytest.mark.asyncio
async def test_list_commit(
    gitlab_client_mock,
    test_params,
    expected_params,
    expected_project_id,
    description,
    metadata,
):
    mock_response = [
        {
            "id": "6104942438c14ec7bd21c6cd5bd995272b3faff6",
            "short_id": "6104942",
            "title": "Test commit",
            "author_name": "John Doe",
            "author_email": "john@example.com",
            "authored_date": "2024-01-01T10:00:00.000Z",
            "committer_name": "John Doe",
            "committer_email": "john@example.com",
            "committed_date": "2024-01-01T10:00:00.000Z",
            "created_at": "2024-01-01T10:00:00.000Z",
            "message": "Test commit message",
            "parent_ids": ["parent123"],
            "web_url": "https://gitlab.example.com/project/-/commit/6104942438c14ec7bd21c6cd5bd995272b3faff6",
        }
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)
    tool = ListCommits(description=description, metadata=metadata)
    result = await tool._arun(**test_params)
    gitlab_client_mock.aget.assert_called_once_with(
        path=f"/api/v4/projects/{expected_project_id}/repository/commits",
        params=expected_params,
        parse_json=False,
    )
    result_data = json.loads(result)
    assert "commits" in result_data
    assert result_data["commits"] == mock_response


@pytest.mark.asyncio
async def test_list_commits_success(gitlab_client_mock, metadata):
    sample_commits = [
        {
            "id": "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "short_id": "c34bb66f",
            "created_at": "2025-04-29T11:35:36.000+02:00",
            "title": "Test Commit",
            "message": "Test Commit message",
            "author_name": "Author Name",
        },
        {
            "id": "4a6c9bc90b6a6c7c96386a74dbade737810bfa78",
            "short_id": "4a6c9bc",
            "created_at": "2025-04-28T10:30:22.000+02:00",
            "title": "Test Commit 2",
            "message": "Test Commit message 2",
            "author_name": "Author Name 2",
        },
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=sample_commits)

    tool = ListCommits(description="list commits description", metadata=metadata)

    response = await tool._arun(
        project_id=24,
        ref_name="main",
        path="src/",
        with_stats=True,
        first_parent=True,
    )

    expected_response = json.dumps({"commits": sample_commits})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits",
        params={
            "ref_name": "main",
            "path": "src/",
            "with_stats": True,
            "first_parent": True,
        },
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_list_commits_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Project not found"))

    tool = ListCommits(description="list commits description", metadata=metadata)

    response = await tool._arun(project_id=999)

    expected_response = json.dumps({"error": "Project not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/999/repository/commits",
        params={},
        parse_json=False,
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListCommitsInput(project_id=123),
            "List commits in project 123",
        ),
        (
            ListCommitsInput(url="https://gitlab.com/namespace/project"),
            "List commits in https://gitlab.com/namespace/project",
        ),
    ],
)
def test_list_commits_format_display_message(input_data, expected_message):
    tool = ListCommits(description="list commits description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_get_commit_diff(gitlab_client_mock, metadata):
    diff_data = [
        {
            "diff": "@@ -0,0 +1,3 @@\n+# Test file\n+\n+This is a test file",
            "new_path": "test.md",
            "old_path": "test.md",
            "a_mode": "0",
            "b_mode": "100644",
            "new_file": True,
            "renamed_file": False,
            "deleted_file": False,
        }
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=json.dumps(diff_data))

    tool = GetCommitDiff(description="Read commit diff", metadata=metadata)

    response = await tool._arun(
        project_id=24, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    )

    expected_response = json.dumps({"diff": diff_data})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/diff",
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_get_commit_diff_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Commit not found"))

    tool = GetCommitDiff(description="Read commit diff", metadata=metadata)

    response = await tool._arun(project_id=24, commit_sha="nonexistent")

    expected_response = json.dumps({"error": "Commit not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/nonexistent/diff",
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,commit_sha,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/diff",
        ),
        # Test with URL and matching project_id and commit_sha
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "namespace%2Fproject",
            "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "/api/v4/projects/namespace%2Fproject/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/diff",
        ),
    ],
)
async def test_get_commit_diff_with_url_success(
    url, project_id, commit_sha, expected_path, gitlab_client_mock, metadata
):
    diff_data = [
        {
            "diff": "@@ -0,0 +1,3 @@\n+# Test file\n+\n+This is a test file",
            "new_path": "test.md",
            "old_path": "test.md",
            "a_mode": "0",
            "b_mode": "100644",
            "new_file": True,
            "renamed_file": False,
            "deleted_file": False,
        }
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=json.dumps(diff_data))

    tool = GetCommitDiff(description="Read commit diff", metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id, commit_sha=commit_sha)

    expected_response = json.dumps({"diff": diff_data})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CommitResourceInput(
                project_id=123, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
            ),
            "Get diff for commit c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6 in project 123",
        ),
        (
            CommitResourceInput(
                url="https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
            ),
            "Get diff for commit https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        ),
    ],
)
def test_get_commit_diff_format_display_message(input_data, expected_message):
    tool = GetCommitDiff(description="Read commit diff")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_get_commit_comments(gitlab_client_mock, metadata):
    comments_data = [
        {
            "note": "This is a comment on the commit",
            "author": {
                "id": 1,
                "name": "John Doe",
                "username": "johndoe",
                "state": "active",
                "avatar_url": "https://gitlab.com/uploads/-/system/user/avatar/1/avatar.png",
                "web_url": "https://gitlab.com/johndoe",
            },
            "created_at": "2025-04-29T11:40:26.000+02:00",
            "path": None,
            "line": None,
            "line_type": None,
            "parent_id": None,
        },
        {
            "note": "Another comment on the commit",
            "author": {
                "id": 2,
                "name": "Jane Smith",
                "username": "janesmith",
                "state": "active",
                "avatar_url": "https://gitlab.com/uploads/-/system/user/avatar/2/avatar.png",
                "web_url": "https://gitlab.com/janesmith",
            },
            "created_at": "2025-04-29T12:15:42.000+02:00",
            "path": "src/main.py",
            "line": 42,
            "line_type": "new",
            "parent_id": None,
        },
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=comments_data)

    tool = GetCommitComments(description="Read commit comments", metadata=metadata)

    response = await tool._arun(
        project_id=24, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    )

    expected_response = json.dumps({"comments": comments_data})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/comments",
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_get_commit_comments_error(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Commit not found"))

    tool = GetCommitComments(description="Read commit comments", metadata=metadata)

    response = await tool._arun(project_id=24, commit_sha="nonexistent")

    expected_response = json.dumps({"error": "Commit not found"})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/nonexistent/comments",
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,commit_sha,expected_path",
    [
        # Test with only URL
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/comments",
        ),
        # Test with URL and matching project_id and commit_sha
        (
            "https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "namespace%2Fproject",
            "c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
            "/api/v4/projects/namespace%2Fproject/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/comments",
        ),
    ],
)
async def test_get_commit_comments_with_url_success(
    url, project_id, commit_sha, expected_path, gitlab_client_mock, metadata
):
    comments_data = [
        {
            "note": "This is a comment on the commit",
            "author": {"id": 1, "name": "John Doe", "username": "johndoe"},
            "created_at": "2025-04-29T11:40:26.000+02:00",
        }
    ]
    gitlab_client_mock.aget = AsyncMock(return_value=comments_data)

    tool = GetCommitComments(description="Read commit comments", metadata=metadata)

    response = await tool._arun(url=url, project_id=project_id, commit_sha=commit_sha)

    expected_response = json.dumps({"comments": comments_data})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CommitResourceInput(
                project_id=123, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
            ),
            "Get comments for commit c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6 in project 123",
        ),
        (
            CommitResourceInput(
                url="https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
            ),
            "Get comments for commit https://gitlab.com/namespace/project/-/commit/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6",
        ),
    ],
)
def test_get_commit_comments_format_display_message(input_data, expected_message):
    tool = GetCommitComments(description="Read commit comments")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_create_commit(gitlab_client_mock, metadata, commit_data):
    """Test basic functionality of CreateCommit._arun method."""
    gitlab_client_mock.apost = AsyncMock(return_value=commit_data)

    tool = CreateCommit(metadata=metadata)

    actions = [
        CreateCommitAction(
            action="create",
            file_path="test.txt",
            content="This is a test file",
        ),
        CreateCommitAction(
            action="update",
            file_path="existing.txt",
            content="Updated content",
        ),
    ]

    actions_list = [action.model_dump(exclude_none=True) for action in actions]

    response = await tool._arun(
        project_id=24,
        branch="main",
        commit_message="Test commit message",
        actions=actions,
        author_name="Test User",
        author_email="test@example.com",
    )

    expected_params = {
        "branch": "main",
        "commit_message": "Test commit message",
        "actions": actions_list,
        "author_email": "test@example.com",
        "author_name": "Test User",
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_params,
            "response": commit_data,
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits",
        body=json.dumps(expected_params),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,expected_path",
    [
        (
            "https://gitlab.com/namespace/project",
            None,
            "/api/v4/projects/namespace%2Fproject/repository/commits",
        ),
        (
            "https://gitlab.com/namespace/project",
            "namespace%2Fproject",
            "/api/v4/projects/namespace%2Fproject/repository/commits",
        ),
    ],
)
async def test_create_commit_with_url_success(
    url, project_id, expected_path, gitlab_client_mock, metadata, commit_data
):
    """Test CreateCommit._arun method with URL parameter."""
    gitlab_client_mock.apost = AsyncMock(return_value=commit_data)

    tool = CreateCommit(metadata=metadata)

    actions = [
        CreateCommitAction(
            action="create",
            file_path="test.txt",
            content="This is a test file",
        ),
    ]

    response = await tool._arun(
        url=url,
        project_id=project_id,
        branch="main",
        commit_message="Test commit message",
        actions=actions,
    )

    actions_list = [action.model_dump(exclude_none=True) for action in actions]

    expected_params = {
        "branch": "main",
        "commit_message": "Test commit message",
        "actions": actions_list,
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_params,
            "response": commit_data,
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_path,
        body=json.dumps(expected_params),
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
async def test_create_commit_with_url_error(
    url, project_id, error_contains, gitlab_client_mock, metadata
):
    """Test error handling in CreateCommit._arun method with invalid URL."""
    tool = CreateCommit(metadata=metadata)

    actions = [
        CreateCommitAction(
            action="create",
            file_path="test.txt",
            content="This is a test file",
        ),
    ]

    response = await tool._arun(
        url=url,
        project_id=project_id,
        branch="main",
        commit_message="Test commit message",
        actions=actions,
    )

    response_json = json.loads(response)

    assert "error" in response_json
    assert error_contains in response_json["error"]
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
async def test_create_commit_with_all_optional_params(
    gitlab_client_mock, metadata, commit_data
):
    """Test CreateCommit._arun method with all optional parameters."""
    gitlab_client_mock.apost = AsyncMock(return_value=commit_data)

    tool = CreateCommit(metadata=metadata)

    actions = [
        CreateCommitAction(
            action="create",
            file_path="test.txt",
            content="This is a test file",
        ),
    ]

    response = await tool._arun(
        project_id=24,
        branch="feature-branch",
        commit_message="Test commit message",
        actions=actions,
        start_branch="main",
        start_sha="abcdef1234567890",
        start_project="namespace/another-project",
        author_email="author@example.com",
        author_name="Author Name",
    )

    actions_list = [action.model_dump(exclude_none=True) for action in actions]

    expected_params = {
        "branch": "feature-branch",
        "commit_message": "Test commit message",
        "actions": actions_list,
        "start_branch": "main",
        "start_sha": "abcdef1234567890",
        "start_project": "namespace/another-project",
        "author_email": "author@example.com",
        "author_name": "Author Name",
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_params,
            "response": commit_data,
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits",
        body=json.dumps(expected_params),
    )


@pytest.mark.asyncio
async def test_create_commit_with_multiple_action_types(
    gitlab_client_mock, metadata, commit_data
):
    """Test CreateCommit._arun method with different action types."""
    gitlab_client_mock.apost = AsyncMock(return_value=commit_data)

    tool = CreateCommit(metadata=metadata)

    actions = [
        CreateCommitAction(
            action="create",
            file_path="new_file.txt",
            content="This is a new file",
        ),
        CreateCommitAction(
            action="update",
            file_path="existing_file.txt",
            content="Updated content",
            last_commit_id="previous_commit_sha",
        ),
        CreateCommitAction(
            action="delete",
            file_path="delete_me.txt",
            last_commit_id="delete_commit_sha",
        ),
        CreateCommitAction(
            action="move",
            file_path="new_path.txt",
            previous_path="old_path.txt",
        ),
        CreateCommitAction(
            action="chmod",
            file_path="script.sh",
            execute_filemode=True,
        ),
    ]

    response = await tool._arun(
        project_id=24,
        branch="main",
        commit_message="Multiple actions commit",
        actions=actions,
    )

    actions_list = [action.model_dump(exclude_none=True) for action in actions[0:-1]]

    expected_params = {
        "branch": "main",
        "commit_message": "Multiple actions commit",
        "actions": actions_list,
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_params,
            "response": commit_data,
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits",
        body=json.dumps(expected_params),
    )


@pytest.mark.asyncio
async def test_create_commit_exception(gitlab_client_mock, metadata):
    """Test exception handling in CreateCommit._arun method."""
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))

    tool = CreateCommit(metadata=metadata)

    actions = [
        CreateCommitAction(
            action="create",
            file_path="test.txt",
            content="This is a test file",
        ),
    ]

    response = await tool._arun(
        project_id=24,
        branch="main",
        commit_message="Test commit message",
        actions=actions,
    )

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateCommitInput(
                project_id=24,
                branch="main",
                commit_message="Test commit",
                actions=[
                    CreateCommitAction(
                        action="create",
                        file_path="test.txt",
                        content="This is a test file",
                    ),
                ],
            ),
            "Create commit in project 24 with 1 file action (create)",
        ),
        (
            CreateCommitInput(
                project_id=24,
                branch="main",
                commit_message="Test commit",
                actions=[
                    CreateCommitAction(
                        action="create",
                        file_path="test1.txt",
                        content="This is test file 1",
                    ),
                    CreateCommitAction(
                        action="update",
                        file_path="test2.txt",
                        content="Updated content",
                    ),
                    CreateCommitAction(
                        action="delete",
                        file_path="test3.txt",
                    ),
                ],
            ),
            "Create commit in project 24 with 3 file actions (create, update, delete)",
        ),
        (
            CreateCommitInput(
                url="https://gitlab.com/namespace/project",
                branch="main",
                commit_message="Test commit",
                actions=[
                    CreateCommitAction(
                        action="create",
                        file_path="test.txt",
                        content="This is a test file",
                    ),
                ],
            ),
            "Create commit in https://gitlab.com/namespace/project with 1 file action (create)",
        ),
    ],
)
def test_create_commit_format_display_message(input_data, expected_message):
    """Test the format_display_message method of CreateCommit."""
    tool = CreateCommit(description="Create commit")
    message = tool.format_display_message(input_data)
    assert message == expected_message


# Tests for DiffExclusionPolicy integration
@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
async def test_get_commit_diff_with_diff_exclusion_policy_enabled(
    mock_diff_feature_flag, mock_file_feature_flag, gitlab_client_mock, metadata
):
    """Test GetCommitDiff applies DiffExclusionPolicy when feature flag is enabled."""
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True

    # Mock diff data with both allowed and excluded files
    diff_data = [
        {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        },
        {
            "old_path": "app.log",
            "new_path": "app.log",
            "diff": "@@ -1,3 +1,3 @@\n-old log\n+new log",
        },
        {
            "old_path": "secrets/api_key.txt",
            "new_path": "secrets/api_key.txt",
            "diff": "@@ -1,3 +1,3 @@\n-old key\n+new key",
        },
        {
            "old_path": "node_modules/react/index.js",
            "new_path": "node_modules/react/index.js",
            "diff": "@@ -1,3 +1,3 @@\n-old react\n+new react",
        },
    ]

    gitlab_client_mock.aget = AsyncMock(return_value=json.dumps(diff_data))

    tool = GetCommitDiff(description="Read commit diff", metadata=metadata)

    response = await tool._arun(
        project_id=24, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    )

    response_data = json.loads(response)

    # Only the allowed diff should remain (src/main.py)
    assert "diff" in response_data
    assert len(response_data["diff"]) == 1

    # Verify the remaining diff is the allowed one
    remaining_diff = response_data["diff"][0]
    assert remaining_diff["old_path"] == "src/main.py"
    assert remaining_diff["new_path"] == "src/main.py"

    # Check for excluded_files and excluded_reason when there are excluded files
    assert "excluded_files" in response_data
    assert "excluded_reason" in response_data

    # Verify excluded files list contains the expected files
    expected_excluded_files = [
        "app.log",
        "secrets/api_key.txt",
        "node_modules/react/index.js",
    ]
    assert set(response_data["excluded_files"]) == set(expected_excluded_files)

    # Verify excluded_reason contains the expected message format
    assert (
        "Files excluded due to policy, continue without files:"
        in response_data["excluded_reason"]
    )
    for excluded_file in expected_excluded_files:
        assert excluded_file in response_data["excluded_reason"]

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/diff",
        parse_json=False,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled")
@patch("duo_workflow_service.policies.diff_exclusion_policy.is_feature_enabled")
async def test_get_commit_diff_with_no_excluded_files(
    mock_diff_feature_flag, mock_file_feature_flag, gitlab_client_mock, metadata
):
    """Test GetCommitDiff does not include excluded_files/excluded_reason when no files are excluded."""
    mock_diff_feature_flag.return_value = True
    mock_file_feature_flag.return_value = True

    # Mock diff data with only allowed files
    diff_data = [
        {
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "diff": "@@ -1,3 +1,3 @@\n-old content\n+new content",
        },
        {
            "old_path": "src/utils.py",
            "new_path": "src/utils.py",
            "diff": "@@ -1,3 +1,3 @@\n-old utils\n+new utils",
        },
    ]

    gitlab_client_mock.aget = AsyncMock(return_value=json.dumps(diff_data))

    tool = GetCommitDiff(description="Read commit diff", metadata=metadata)

    response = await tool._arun(
        project_id=24, commit_sha="c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6"
    )

    response_data = json.loads(response)

    # All diffs should remain
    assert "diff" in response_data
    assert len(response_data["diff"]) == 2

    # Should not include excluded_files or excluded_reason when no files are excluded
    assert "excluded_files" not in response_data
    assert "excluded_reason" not in response_data

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/24/repository/commits/c34bb66f7a5e3a45b5e2d70edd9be12d64855cd6/diff",
        parse_json=False,
    )
