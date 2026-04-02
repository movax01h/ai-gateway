import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    ListAllMergeRequestNotes,
    MergeRequestResourceInput,
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
        body=json.dumps(response_data),
        headers={"content-type": "application/json", "X-Next-Page": ""},
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
async def test_list_all_merge_request_notes(gitlab_client_mock, metadata):
    notes = [{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]
    mock_response = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(notes),
        headers={"X-Next-Page": ""},
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListAllMergeRequestNotes(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"notes": notes})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/notes",
        params={"page": "1", "per_page": 100},
        parse_json=False,
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
    notes_data = [{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]
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
        {"notes": [{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={"page": "1", "per_page": 100},
        parse_json=False,
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
async def test_list_all_merge_request_notes_pagination(gitlab_client_mock, metadata):
    """Notes spanning multiple pages are all collected and returned."""
    page1_notes = [{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]
    page2_notes = [{"id": 3, "body": "Note 3"}]

    mock_page1 = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(page1_notes),
        headers={"X-Next-Page": "2"},
    )
    mock_page2 = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(page2_notes),
        headers={"X-Next-Page": ""},
    )
    gitlab_client_mock.aget = AsyncMock(side_effect=[mock_page1, mock_page2])

    tool = ListAllMergeRequestNotes(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"notes": page1_notes + page2_notes})
    assert response == expected_response
    assert gitlab_client_mock.aget.call_count == 2


@pytest.mark.asyncio
async def test_list_all_merge_request_notes_exception(gitlab_client_mock, metadata):
    """Test exception handling in ListAllMergeRequestNotes._arun method."""
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = ListAllMergeRequestNotes(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"error": error_message})
    assert response == expected_response


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
