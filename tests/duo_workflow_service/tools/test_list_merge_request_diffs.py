import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.merge_request import (
    ListMergeRequestDiffs,
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
async def test_list_merge_request_diffs(gitlab_client_mock, metadata):
    mock_response = GitLabHttpResponse(
        status_code=200,
        body="[]",
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListMergeRequestDiffs(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    expected_response = json.dumps({"diffs": []})
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/diffs",
        parse_json=False,
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
    diffs_data = "[]"
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

    expected_response = json.dumps({"diffs": []})
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
async def test_list_merge_request_diffs_exception(gitlab_client_mock, metadata):
    """Test that exceptions from ListMergeRequestDiffs._execute propagate rather than being swallowed."""
    error_message = "API error"
    gitlab_client_mock.aget = AsyncMock(side_effect=Exception(error_message))

    tool = ListMergeRequestDiffs(metadata=metadata)

    with pytest.raises(Exception, match=error_message):
        await tool._arun(project_id=1, merge_request_iid=123)


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


# Tests for DiffExclusionPolicy integration
@pytest.mark.asyncio
async def test_list_merge_request_diffs_with_diff_exclusion_policy_enabled(
    gitlab_client_mock, metadata
):
    """Test ListMergeRequestDiffs applies DiffExclusionPolicy when feature flag is enabled."""
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

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(diff_data),
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListMergeRequestDiffs(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    response_data = json.loads(response)

    # Only the allowed diff should remain (src/main.py)
    assert "diffs" in response_data
    assert len(response_data["diffs"]) == 1

    # Verify the remaining diff is the allowed one
    remaining_diff = response_data["diffs"][0]
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
        path="/api/v4/projects/1/merge_requests/123/diffs",
        parse_json=False,
    )


@pytest.mark.asyncio
async def test_list_merge_request_diffs_with_no_excluded_files(
    gitlab_client_mock, metadata
):
    """Test ListMergeRequestDiffs does not include excluded_files/excluded_reason when no files are excluded."""
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

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=json.dumps(diff_data),
    )
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    tool = ListMergeRequestDiffs(metadata=metadata)

    response = await tool._arun(project_id=1, merge_request_iid=123)

    response_data = json.loads(response)

    # All diffs should remain
    assert "diffs" in response_data
    assert len(response_data["diffs"]) == 2

    # Should not include excluded_files or excluded_reason when no files are excluded
    assert "excluded_files" not in response_data
    assert "excluded_reason" not in response_data

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/merge_requests/123/diffs",
        parse_json=False,
    )
