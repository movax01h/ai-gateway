import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.merge_request import (
    CreateMergeRequest,
    CreateMergeRequestInput,
    CreateMergeRequestNote,
    CreateMergeRequestNoteInput,
    GetMergeRequest,
    ListAllMergeRequestNotes,
    ListMergeRequestDiffs,
    MergeRequestResourceInput,
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


@pytest.fixture
def merge_request_data():
    """Fixture for common merge request data."""
    return {
        "id": 1,
        "title": "Test Merge Request",
        "source_branch": "feature",
        "target_branch": "main",
    }


@pytest.fixture
def note_data():
    return {
        "id": 1,
        "body": "Test note",
        "created_at": "2024-01-01T12:00:00Z",
        "author": {"id": 1, "name": "Test User"},
    }


@pytest.fixture
def gitlab_client_mock():
    return Mock()


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
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
    gitlab_client_mock.aget = AsyncMock(return_value=response_data)
    gitlab_client_mock.apost = AsyncMock(return_value=response_data)
    gitlab_client_mock.aput = AsyncMock(return_value=response_data)

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
    gitlab_client_mock.apost = AsyncMock(
        return_value='{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}'
    )

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
    }

    expected_response = json.dumps(
        {
            "status": "success",
            "data": expected_data,
            "response": '{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}',
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests", body=json.dumps(expected_data)
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
            "status": "success",
            "data": {
                "source_branch": "feature",
                "target_branch": "main",
                "title": "Test Merge Request",
            },
            "response": {
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
async def test_create_merge_request_minimal_params(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(
        return_value='{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}'
    )

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
            "status": "success",
            "data": expected_data,
            "response": '{"id": 1, "title": "New Feature", "source_branch": "feature", "target_branch": "main"}',
        }
    )

    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests", body=json.dumps(expected_data)
    )


@pytest.mark.asyncio
async def test_get_merge_request(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value='{"id": 1, "title": "Test MR", "source_branch": "feature", "target_branch": "main"}'
    )

    tool = GetMergeRequest(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps(
        {
            "merge_request": '{"id": 1, "title": "Test MR", "source_branch": "feature", "target_branch": "main"}'
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123", parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    URL_SUCCESS_CASES,
)
async def test_get_merge_request_with_url_success(
    url,
    project_id,
    merge_request_iid,
    expected_path,
    gitlab_client_mock,
    metadata,
    merge_request_data,
):
    tool = GetMergeRequest(
        description="get merge request description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=merge_request_data,
    )

    expected_response = json.dumps(
        {
            "merge_request": {
                "id": 1,
                "title": "Test Merge Request",
                "source_branch": "feature",
                "target_branch": "main",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_get_merge_request_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = GetMergeRequest(
        description="get merge request description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        # Missing project_id
        (
            {"merge_request_iid": 123},
            "'project_id' must be provided when 'url' is not",
        ),
        # Missing merge_request_iid
        (
            {"project_id": 1},
            "'merge_request_iid' must be provided when 'url' is not",
        ),
        # Missing both project_id and merge_request_iid
        (
            {},
            "'project_id' must be provided when 'url' is not; 'merge_request_iid' must be provided when 'url' is not",
        ),
    ],
)
async def test_get_merge_request_validation(kwargs, expected_error, metadata):
    gitlab_client_mock = metadata["gitlab_client"]
    tool = GetMergeRequest(
        description="get merge request description", metadata=metadata
    )

    response = await tool._arun(**kwargs)
    response_json = json.loads(response)

    assert "error" in response_json
    assert expected_error in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_list_merge_request_diffs(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(return_value='{"diffs": []}')

    tool = ListMergeRequestDiffs(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"diffs": '{"diffs": []}'})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/diffs", parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_path",
    [
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/diffs",
        ),
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/diffs",
        ),
    ],
)
async def test_list_merge_request_diffs_with_url_success(
    url, project_id, merge_request_iid, expected_path, gitlab_client_mock, metadata
):
    diffs_data = '{"diffs": []}'
    tool = ListMergeRequestDiffs(
        description="list merge request diffs description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=diffs_data,
    )

    expected_response = json.dumps({"diffs": '{"diffs": []}'})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_list_merge_request_diffs_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = ListMergeRequestDiffs(
        description="list merge request diffs description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
    )


@pytest.mark.asyncio
async def test_create_merge_request_note(gitlab_client_mock, metadata):
    gitlab_client_mock.apost = AsyncMock(return_value='{"id": 1, "body": "Test note"}')

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123, body="Test note")

    expected_response = json.dumps(
        {
            "status": "success",
            "body": "Test note",
            "response": '{"id": 1, "body": "Test note"}',
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body='{"body": "Test note"}',
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
            "status": "success",
            "body": "Test note",
            "response": {
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
        "/merge",
        "/close",
        "/label ~bug",
        "/assign @user",
        "/milestone %v1.0",
        "/remove_source_branch",
        "/target_branch main",
        "/title Update title",
        "/board_move ~doing",
        "/copy_metadata from !123",
        "This is a multi-line note\n/merge",
        "Line 1\n/close\nLine 3",
        "/MErGE",
    ],
)
async def test_create_merge_request_note_blocks_quick_actions(
    note, gitlab_client_mock, metadata
):
    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "body": note}
    )

    expected_response = json.dumps(
        {
            "status": "error",
            "message": """Notes containing GitLab quick actions are not allowed. Quick actions are text-based shortcuts for common GitLab actions.
                                  They are commands that are on their own line and start with a backslash. Examples include /merge, /approve, /close, etc.""",
        }
    )

    assert response == expected_response
    gitlab_client_mock.apost.assert_not_called()


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
    gitlab_client_mock.apost = AsyncMock(
        side_effect=['{"id": 1, "body": "' + note + '"}']
    )

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "body": note}
    )

    expected_response = json.dumps(
        {
            "status": "success",
            "body": note,
            "response": '{"id": 1, "body": "' + note + '"}',
        }
    )

    assert response == expected_response
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        body=json.dumps({"body": note}),
    )


@pytest.mark.asyncio
async def test_list_all_merge_request_notes(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = AsyncMock(
        return_value='[{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]'
    )

    tool = ListAllMergeRequestNotes(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps(
        {"notes": '[{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]'}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes", parse_json=False
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
async def test_list_all_merge_request_notes_with_url_success(
    url, project_id, merge_request_iid, expected_path, gitlab_client_mock, metadata
):
    notes_data = '[{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]'
    tool = ListAllMergeRequestNotes(
        description="list merge request notes description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=notes_data,
    )

    expected_response = json.dumps(
        {"notes": '[{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]'}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_list_all_merge_request_notes_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = ListAllMergeRequestNotes(
        description="list merge request notes description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
    )


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
    gitlab_client_mock.aput = AsyncMock(return_value=update_data)
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


@pytest.mark.asyncio
async def test_get_merge_request_exception(gitlab_client_mock, metadata):
    """Test exception handling in GetMergeRequest._arun method."""
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = GetMergeRequest(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.asyncio
async def test_list_merge_request_diffs_exception(gitlab_client_mock, metadata):
    """Test exception handling in ListMergeRequestDiffs._arun method."""
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = ListMergeRequestDiffs(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.asyncio
async def test_create_merge_request_note_exception(gitlab_client_mock, metadata):
    """Test exception handling in CreateMergeRequestNote._arun method."""
    error_message = "API error"
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception(error_message))

    tool = CreateMergeRequestNote(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123, body="Test note")

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.asyncio
async def test_list_all_merge_request_notes_exception(gitlab_client_mock, metadata):
    """Test exception handling in ListAllMergeRequestNotes._arun method."""
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = ListAllMergeRequestNotes(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


@pytest.mark.asyncio
async def test_update_merge_request_exception(gitlab_client_mock, metadata):
    """Test exception handling in UpdateMergeRequest._arun method."""
    error_message = "API error"
    gitlab_client_mock.aput = AsyncMock(side_effect=Exception(error_message))

    tool = UpdateMergeRequest(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123, title="Updated MR")

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


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            MergeRequestResourceInput(project_id=42, merge_request_iid=123),
            "Read merge request !123 in project 42",
        ),
        (
            MergeRequestResourceInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "Read merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_get_merge_request_format_display_message(input_data, expected_message):
    tool = GetMergeRequest(description="Get merge request description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            MergeRequestResourceInput(project_id=42, merge_request_iid=123),
            "View changes in merge request !123 in project 42",
        ),
        (
            MergeRequestResourceInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "View changes in merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_list_merge_request_diffs_format_display_message(input_data, expected_message):
    tool = ListMergeRequestDiffs(description="List merge request diffs")
    message = tool.format_display_message(input_data)
    assert message == expected_message


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
    ],
)
def test_create_merge_request_note_format_display_message(input_data, expected_message):
    tool = CreateMergeRequestNote(description="Create merge request note")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            MergeRequestResourceInput(project_id=42, merge_request_iid=123),
            "Read comments on merge request !123 in project 42",
        ),
        (
            MergeRequestResourceInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42"
            ),
            "Read comments on merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_list_all_merge_request_notes_format_display_message(
    input_data, expected_message
):
    tool = ListAllMergeRequestNotes(description="List merge request notes")
    message = tool.format_display_message(input_data)
    assert message == expected_message


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
