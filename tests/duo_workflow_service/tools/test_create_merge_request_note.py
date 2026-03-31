import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    CreateMergeRequestNote,
    CreateMergeRequestNoteInput,
)

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


@pytest.fixture(name="note_data")
def note_data_fixture():
    return {
        "id": 1,
        "body": "Test note",
        "created_at": "2024-01-01T12:00:00Z",
        "author": {"id": 1, "name": "Test User"},
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
async def test_create_merge_request_note(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={"id": 1, "body": "Test note"},
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123, body="Test note")

    expected_response = json.dumps(
        {
            "created_merge_request_note": {"id": 1, "body": "Test note"},
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body='{"body": "Test note"}',
    )


@pytest.mark.asyncio
async def test_create_merge_request_note_with_note_id_reply(
    gitlab_client_mock, metadata
):
    """Test creating a reply to an existing note using note_id."""
    discussions_response = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(
            [{"id": "abc123", "notes": [{"id": 1, "body": "Original note"}]}]
        ),
    )

    reply_response = GitLabHttpResponse(
        status_code=200,
        body={
            "id": 2,
            "body": "This is a reply",
            "discussion_id": "gid://gitlab/Discussion/789",
        },
    )

    gitlab_client_mock.aget = AsyncMock(return_value=discussions_response)
    gitlab_client_mock.apost = AsyncMock(return_value=reply_response)

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=123,
        body="This is a reply",
        note_id=1,
    )

    expected_response = json.dumps(
        {
            "created_merge_request_note": {
                "id": 2,
                "body": "This is a reply",
                "discussion_id": "gid://gitlab/Discussion/789",
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/discussions",
        params={"page": "1", "per_page": 100},
        parse_json=False,
    )

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/discussions/abc123/notes",
        body=json.dumps({"body": "This is a reply"}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    [
        # Modify paths for notes endpoints
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/notes",
        ),
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/notes",
        ),
    ],
)
async def test_create_merge_request_note_with_url_success(
    url,
    project_id,
    merge_request_iid,
    expected_path,
    gitlab_client_mock,
    metadata,
    note_data,
):
    tool = CreateMergeRequestNote(
        description="create merge request note description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=note_data,
        body="Test note",
    )

    expected_response = json.dumps(
        {
            "created_merge_request_note": {
                "id": 1,
                "body": "Test note",
                "created_at": "2024-01-01T12:00:00Z",
                "author": {"id": 1, "name": "Test User"},
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_path,
        body=json.dumps({"body": "Test note"}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    [
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/discussions/abc123/notes",
        ),
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/discussions/abc123/notes",
        ),
    ],
)
async def test_create_merge_request_note_with_url_and_note_id_success(
    url,
    project_id,
    merge_request_iid,
    expected_path,
    gitlab_client_mock,
    metadata,
):
    """Test creating a reply to an existing note via URL with note_id."""
    discussions_response = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(
            [{"id": "abc123", "notes": [{"id": 1, "body": "Original note"}]}]
        ),
    )

    # Mock for creating the reply
    reply_data = {
        "id": 2,
        "body": "This is a reply",
        "created_at": "2024-01-01T13:00:00Z",
        "author": {"id": 1, "name": "Test User"},
        "discussion_id": "gid://gitlab/Discussion/789",
    }

    reply_response = GitLabHttpResponse(
        status_code=200,
        body=reply_data,
        headers={"content-type": "application/json"},
    )

    gitlab_client_mock.aget = AsyncMock(return_value=discussions_response)
    gitlab_client_mock.apost = AsyncMock(return_value=reply_response)

    tool = CreateMergeRequestNote(
        description="create merge request note description", metadata=metadata
    )

    response = await tool._arun(
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        body="This is a reply",
        note_id=1,
    )

    expected_response = json.dumps(
        {
            "created_merge_request_note": {
                "id": 2,
                "body": "This is a reply",
                "created_at": "2024-01-01T13:00:00Z",
                "author": {"id": 1, "name": "Test User"},
                "discussion_id": "gid://gitlab/Discussion/789",
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_path,
        body=json.dumps({"body": "This is a reply"}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_create_merge_request_note_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = CreateMergeRequestNote(
        description="create merge request note description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        body="Test note",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "note",
    [
        "This is a regular note",
        "This note talks about /merge in the middle",
        "https://gitlab.com",
        "gitlab-org/gitlab",
        "URL: https://example.com/merge",
        "Text with slash/merge in middle",
        "Line 1\nLine 2\nLine 3",
        "Discussion about\nmerge\nand \nclose\n without slashes",
    ],
)
async def test_create_merge_request_note_allows_regular_notes(
    note, gitlab_client_mock, metadata
):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={"id": 1, "body": note},
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "body": note}
    )

    expected_response = json.dumps(
        {
            "created_merge_request_note": {"id": 1, "body": note},
        }
    )

    assert response == expected_response
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body=json.dumps({"body": note}),
    )


# ---- CreateMergeRequestNote internal parameter tests ----


@pytest.mark.asyncio
async def test_create_merge_request_note_with_internal_true(
    gitlab_client_mock, metadata
):
    """Test creating an internal note by setting internal=True."""
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={"id": 1, "body": "Internal note", "internal": True},
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(
        project_id=1, merge_request_iid=123, body="Internal note", internal=True
    )

    expected_response = json.dumps(
        {
            "created_merge_request_note": {
                "id": 1,
                "body": "Internal note",
                "internal": True,
            },
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body=json.dumps({"body": "Internal note", "internal": True}),
    )


@pytest.mark.asyncio
async def test_create_merge_request_note_with_internal_false(
    gitlab_client_mock, metadata
):
    """Test creating a public note by setting internal=False (default)."""
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={"id": 1, "body": "Public note"},
    )
    gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(
        project_id=1, merge_request_iid=123, body="Public note", internal=False
    )

    expected_response = json.dumps(
        {
            "created_merge_request_note": {"id": 1, "body": "Public note"},
        }
    )
    assert response == expected_response

    # internal should not be in payload when False
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body=json.dumps({"body": "Public note"}),
    )


@pytest.mark.asyncio
async def test_create_merge_request_note_exception(gitlab_client_mock, metadata):
    """Test exception handling in CreateMergeRequestNote._arun method."""
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123, body="Test note")

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateMergeRequestNoteInput(
                project_id=42,
                merge_request_iid=123,
                body="This is a note on the merge request",
            ),
            "Add comment to merge request !123 in project 42",
        ),
        (
            CreateMergeRequestNoteInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                body="This is a note on the merge request",
            ),
            "Add comment to merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
        (
            CreateMergeRequestNoteInput(
                project_id=42,
                merge_request_iid=123,
                body="This is a reply to a comment",
                note_id=1,
            ),
            "Add comment to merge request !123 in project 42",
        ),
        (
            CreateMergeRequestNoteInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                body="This is a reply to a comment",
                note_id=1,
            ),
            "Add comment to merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_create_merge_request_note_format_display_message(input_data, expected_message):
    tool = CreateMergeRequestNote(description="Create merge request note")
    message = tool.format_display_message(input_data)
    assert message == expected_message
