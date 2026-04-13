import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    GetMergeRequest,
    MergeRequestResourceInput,
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
async def test_get_merge_request(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body={
            "id": 1,
            "title": "Test MR",
            "source_branch": "feature",
            "target_branch": "main",
        },
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = GetMergeRequest(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps(
        {
            "merge_request": {
                "id": 1,
                "title": "Test MR",
                "source_branch": "feature",
                "target_branch": "main",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123",
        parse_json=False,
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

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(**kwargs)

    assert expected_error in str(exc_info.value)
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_get_merge_request_exception(gitlab_client_mock, metadata):
    """Test that exceptions from GetMergeRequest._execute propagate rather than being swallowed."""
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = GetMergeRequest(metadata=metadata)

    with pytest.raises(Exception, match=error_message):
        await tool._arun(project_id=1, merge_request_iid=123)


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
