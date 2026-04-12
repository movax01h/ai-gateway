import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    CreateMergeRequestDiffNote,
    CreateMergeRequestDiffNoteInput,
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

DIFF_REFS = {
    "base_sha": "abc123",
    "head_sha": "def456",
    "start_sha": "ghi789",
}


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="project_mock")
def project_mock_fixture():
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


@pytest.fixture(name="mr_response")
def mr_response_fixture():
    """Mock MR GET response containing diff_refs."""
    return GitLabHttpResponse(
        status_code=200,
        body=json.dumps({"id": 9, "diff_refs": DIFF_REFS}),
    )


@pytest.fixture(name="discussion_response")
def discussion_response_fixture():
    """Mock discussion POST response."""
    return GitLabHttpResponse(
        status_code=200,
        body={
            "id": "disc123",
            "notes": [{"id": 1, "body": "suggestion note", "type": "DiffNote"}],
        },
    )


@pytest.mark.asyncio
async def test_create_diff_note_with_new_line(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting an inline note on an added line (new_line only)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="```suggestion:-0+0\nfixed code\n```",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=42,
    )

    result = json.loads(response)
    assert "created_diff_note" in result
    assert result["created_diff_note"]["id"] == "disc123"

    # Verify the MR was fetched for diff_refs
    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/9",
        parse_json=False,
    )

    # Verify the discussion was created with correct payload
    expected_payload = {
        "body": "```suggestion:-0+0\nfixed code\n```",
        "position": {
            "position_type": "text",
            "base_sha": "abc123",
            "head_sha": "def456",
            "start_sha": "ghi789",
            "old_path": "src/main.py",
            "new_path": "src/main.py",
            "new_line": 42,
        },
    }
    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/9/discussions",
        body=json.dumps(expected_payload),
    )


@pytest.mark.asyncio
async def test_create_diff_note_with_old_line(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting an inline note on a deleted line (old_line only)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="This line should not be removed",
        old_path="src/main.py",
        new_path="src/main.py",
        old_line=10,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    call_args = gitlab_client_mock.apost.call_args
    payload = json.loads(call_args.kwargs["body"])
    assert payload["position"]["old_line"] == 10
    assert "new_line" not in payload["position"]


@pytest.mark.asyncio
async def test_create_diff_note_with_both_lines(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting an inline note on an unchanged/context line (both old_line and new_line)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Comment on context line",
        old_path="src/main.py",
        new_path="src/main.py",
        old_line=5,
        new_line=5,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    call_args = gitlab_client_mock.apost.call_args
    payload = json.loads(call_args.kwargs["body"])
    assert payload["position"]["old_line"] == 5
    assert payload["position"]["new_line"] == 5


@pytest.mark.asyncio
async def test_create_diff_note_with_renamed_file(
    gitlab_client_mock, metadata, mr_response, discussion_response
):
    """Test posting a note where old_path differs from new_path (file rename)."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Note on renamed file",
        old_path="src/old_name.py",
        new_path="src/new_name.py",
        new_line=1,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    call_args = gitlab_client_mock.apost.call_args
    payload = json.loads(call_args.kwargs["body"])
    assert payload["position"]["old_path"] == "src/old_name.py"
    assert payload["position"]["new_path"] == "src/new_name.py"


@pytest.mark.asyncio
async def test_create_diff_note_no_line_provided(gitlab_client_mock, metadata):
    """Test error when neither old_line nor new_line is provided."""
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Missing line info",
        old_path="src/main.py",
        new_path="src/main.py",
    )

    result = json.loads(response)
    assert "error" in result
    assert "old_line or new_line" in result["error"]


@pytest.mark.asyncio
async def test_create_diff_note_diff_refs_fetch_failure(gitlab_client_mock, metadata):
    """Test error when fetching diff_refs fails (non-JSON body triggers JSONDecodeError)."""
    mr_error_response = GitLabHttpResponse(
        status_code=404,
        body="Not Found",
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mr_error_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=999,
        body="Note",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=1,
    )

    result = json.loads(response)
    assert "error" in result


@pytest.mark.asyncio
async def test_create_diff_note_diff_refs_missing(gitlab_client_mock, metadata):
    """Test error when MR response has no diff_refs field."""
    mr_response_no_refs = GitLabHttpResponse(
        status_code=200,
        body=json.dumps({"id": 9}),
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response_no_refs)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Note",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=1,
    )

    result = json.loads(response)
    assert "error" in result


@pytest.mark.asyncio
async def test_create_diff_note_post_exception(
    gitlab_client_mock, metadata, mr_response
):
    """Test exception handling when the discussion POST fails."""
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(side_effect=Exception("API error"))

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        project_id=1,
        merge_request_iid=9,
        body="Note",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=1,
    )

    result = json.loads(response)
    assert result == {"error": "API error"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_create_diff_note_with_url_error(
    url, project_id, merge_request_iid, error_contains, gitlab_client_mock, metadata
):
    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        body="Note",
        old_path="src/main.py",
        new_path="src/main.py",
        new_line=1,
    )

    result = json.loads(response)
    assert "error" in result
    assert error_contains in result["error"]

    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.apost.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,merge_request_iid,expected_mr_path,expected_disc_path",
    [
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123",
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/discussions",
        ),
        (
            "https://gitlab.com/namespace/project/-/merge_requests/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/merge_requests/123",
            "/api/v4/projects/namespace%2Fproject/merge_requests/123/discussions",
        ),
    ],
)
async def test_create_diff_note_with_url_success(
    url,
    project_id,
    merge_request_iid,
    expected_mr_path,
    expected_disc_path,
    gitlab_client_mock,
    metadata,
    mr_response,
    discussion_response,
):
    gitlab_client_mock.aget = AsyncMock(return_value=mr_response)
    gitlab_client_mock.apost = AsyncMock(return_value=discussion_response)

    tool = CreateMergeRequestDiffNote(metadata=metadata)

    response = await tool._arun(
        url=url,
        project_id=project_id,
        merge_request_iid=merge_request_iid,
        body="Inline note",
        old_path="file.py",
        new_path="file.py",
        new_line=10,
    )

    result = json.loads(response)
    assert "created_diff_note" in result

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_mr_path,
        parse_json=False,
    )
    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_disc_path,
        body=json.dumps(
            {
                "body": "Inline note",
                "position": {
                    "position_type": "text",
                    "base_sha": "abc123",
                    "head_sha": "def456",
                    "start_sha": "ghi789",
                    "old_path": "file.py",
                    "new_path": "file.py",
                    "new_line": 10,
                },
            }
        ),
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateMergeRequestDiffNoteInput(
                project_id=42,
                merge_request_iid=9,
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                new_line=10,
            ),
            "Add inline diff note to src/main.py:new_line=10 in merge request !9 in project 42",
        ),
        (
            CreateMergeRequestDiffNoteInput(
                project_id=42,
                merge_request_iid=9,
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                old_line=5,
            ),
            "Add inline diff note to src/main.py:old_line=5 in merge request !9 in project 42",
        ),
        (
            CreateMergeRequestDiffNoteInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                new_line=10,
            ),
            "Add inline diff note to src/main.py:new_line=10 in merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
        (
            CreateMergeRequestDiffNoteInput(
                url="https://gitlab.com/namespace/project/-/merge_requests/42",
                body="suggestion",
                old_path="src/main.py",
                new_path="src/main.py",
                old_line=5,
            ),
            "Add inline diff note to src/main.py:old_line=5 in merge request https://gitlab.com/namespace/project/-/merge_requests/42",
        ),
    ],
)
def test_create_diff_note_format_display_message(input_data, expected_message):
    tool = CreateMergeRequestDiffNote(description="Create diff note")
    message = tool.format_display_message(input_data)
    assert message == expected_message
