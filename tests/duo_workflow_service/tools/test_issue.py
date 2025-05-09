import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.issue import (
    CreateIssue,
    CreateIssueInput,
    CreateIssueNote,
    CreateIssueNoteInput,
    GetIssue,
    GetIssueNote,
    GetIssueNoteInput,
    IssueResourceInput,
    ListIssueNotes,
    ListIssueNotesInput,
    ListIssues,
    ListIssuesInput,
    UpdateIssue,
    UpdateIssueInput,
)

# Common URL test parameters
URL_SUCCESS_CASES = [
    # Test with only URL
    (
        "https://gitlab.com/namespace/project/-/issues/123",
        None,
        None,
        "/api/v4/projects/namespace%2Fproject/issues/123",
    ),
    # Test with URL and matching project_id and issue_iid
    (
        "https://gitlab.com/namespace/project/-/issues/123",
        "namespace%2Fproject",
        123,
        "/api/v4/projects/namespace%2Fproject/issues/123",
    ),
]

URL_ERROR_CASES = [
    # URL and project_id both given, but don't match
    (
        "https://gitlab.com/namespace/project/-/issues/123",
        "different%2Fproject",
        123,
        "Project ID mismatch",
    ),
    # URL and issue_iid both given, but don't match
    (
        "https://gitlab.com/namespace/project/-/issues/123",
        "namespace%2Fproject",
        456,
        "Issue ID mismatch",
    ),
    # URL given isn't an issue URL (it's just a project URL)
    (
        "https://gitlab.com/namespace/project",
        None,
        None,
        "Failed to parse URL",
    ),
]


@pytest.fixture
def issue_data():
    """Fixture for common issue data"""
    return {
        "id": 1,
        "title": "Test Issue",
        "description": "This is a test issue",
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
def issues_list_data():
    return [
        {"id": 1, "title": "Issue 1"},
        {"id": 2, "title": "Issue 2"},
    ]


@pytest.fixture
def gitlab_client_mock():
    mock = Mock()
    return mock


@pytest.fixture
def metadata(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture
def issue_tool_setup():
    """Fixture that provides a mock GitLab client and metadata with standard issue data"""
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.return_value = {
        "id": 1,
        "title": "Test Issue",
        "description": "This is a test issue",
    }
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }
    return gitlab_client_mock, metadata


# Helper functions for common test patterns
async def tool_url_success_response(
    tool,
    url,
    project_id,
    issue_iid,
    gitlab_client_mock,
    response_data,
    **kwargs,
):
    gitlab_client_mock.aget = AsyncMock(return_value=response_data)
    gitlab_client_mock.apost = AsyncMock(return_value=response_data)
    gitlab_client_mock.aput = AsyncMock(return_value=response_data)

    response = await tool._arun(
        url=url, project_id=project_id, issue_iid=issue_iid, **kwargs
    )

    return response


async def assert_tool_url_error(
    tool, url, project_id, issue_iid, error_contains, gitlab_client_mock, **kwargs
):
    response = await tool._arun(
        url=url, project_id=project_id, issue_iid=issue_iid, **kwargs
    )

    error_response = json.loads(response)
    assert "error" in error_response
    assert error_contains in error_response["error"]

    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.apost.assert_not_called()
    gitlab_client_mock.aput.assert_not_called()

    return response


@pytest.mark.asyncio
async def test_create_issue(gitlab_client_mock, metadata, issue_data):
    gitlab_client_mock.apost = AsyncMock(return_value=issue_data)

    tool = CreateIssue(description="created issue description", metadata=metadata)

    response = await tool._arun(
        project_id=1,
        title="Test Issue",
        description="This is a test issue",
        labels="bug,urgent",
        assignee_ids=[10, 11],
        milestone_id=5,
        due_date="2023-12-31",
    )

    expected_response = json.dumps(
        {
            "created_issue": {
                "id": 1,
                "title": "Test Issue",
                "description": "This is a test issue",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path="/api/v4/projects/1/issues",
        body=json.dumps(
            {
                "title": "Test Issue",
                "description": "This is a test issue",
                "labels": "bug,urgent",
                "assignee_ids": [10, 11],
                "milestone_id": 5,
                "due_date": "2023-12-31",
            }
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,expected_path",
    [
        # Test with only URL for project
        (
            "https://gitlab.com/namespace/project",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/issues",
        ),
        # Test with URL and matching project_id
        (
            "https://gitlab.com/namespace/project",
            "namespace%2Fproject",
            None,
            "/api/v4/projects/namespace%2Fproject/issues",
        ),
    ],
)
async def test_create_issue_with_url_success(
    url, project_id, issue_iid, expected_path, gitlab_client_mock, metadata, issue_data
):
    tool = CreateIssue(description="create issue description", metadata=metadata)

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=issue_data,
        title="Test Issue",
        description="This is a test issue",
    )

    expected_response = json.dumps(
        {
            "created_issue": {
                "id": 1,
                "title": "Test Issue",
                "description": "This is a test issue",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.apost.assert_called_once_with(
        path=expected_path,
        body=json.dumps(
            {
                "title": "Test Issue",
                "description": "This is a test issue",
            }
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,error_contains",
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
async def test_create_issue_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = CreateIssue(description="create issue description", metadata=metadata)

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        title="Test Issue",
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateIssueInput(
                project_id=123,
                title="New Bug Report",
                description="This is a test issue",
                labels=None,
                assignee_ids=None,
                confidential=None,
                due_date=None,
                issue_type=None,
            ),
            "Create issue 'New Bug Report' in project 123",
        ),
        (
            CreateIssueInput(
                url="https://gitlab.com/namespace/project",
                title="New Bug Report",
                description="This is a test issue",
                labels=None,
                assignee_ids=None,
                confidential=None,
                due_date=None,
                issue_type=None,
            ),
            "Create issue 'New Bug Report' in https://gitlab.com/namespace/project",
        ),
    ],
)
def test_create_issue_format_display_message(input_data, expected_message):
    tool = CreateIssue(description="Create issue description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,labels,milestone,scope,search,expected_params",
    [
        (None, None, None, None, None, {}),
        (
            "opened",
            "bug",
            "v1.0",
            "all",
            "important",
            {
                "state": "opened",
                "labels": "bug",
                "milestone": "v1.0",
                "scope": "all",
                "search": "important",
            },
        ),
    ],
)
async def test_list_issues(
    state,
    labels,
    milestone,
    scope,
    search,
    expected_params,
    gitlab_client_mock,
    metadata,
    issues_list_data,
):
    gitlab_client_mock.aget = AsyncMock(return_value=issues_list_data)

    tool = ListIssues(description="listed issue description", metadata=metadata)

    response = await tool._arun(
        project_id=1,
        state=state,
        labels=labels,
        milestone=milestone,
        scope=scope,
        search=search,
    )

    expected_response = json.dumps(
        {"issues": [{"id": 1, "title": "Issue 1"}, {"id": 2, "title": "Issue 2"}]}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/issues", params=expected_params, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,expected_path",
    [
        # Test with only URL for project
        (
            "https://gitlab.com/namespace/project",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/issues",
        ),
        # Test with URL and matching project_id
        (
            "https://gitlab.com/namespace/project",
            "namespace%2Fproject",
            None,
            "/api/v4/projects/namespace%2Fproject/issues",
        ),
    ],
)
async def test_list_issues_with_url_success(
    url,
    project_id,
    issue_iid,
    expected_path,
    gitlab_client_mock,
    metadata,
    issues_list_data,
):
    tool = ListIssues(description="list issues description", metadata=metadata)

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=issues_list_data,
        state="opened",
    )

    expected_response = json.dumps(
        {"issues": [{"id": 1, "title": "Issue 1"}, {"id": 2, "title": "Issue 2"}]}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={"state": "opened"},
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,error_contains",
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
async def test_list_issues_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = ListIssues(description="list issues description", metadata=metadata)

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListIssuesInput(
                project_id=123,
                assignee_id=None,
                assignee_usernames=None,
                author_id=None,
                author_username=None,
                confidential=None,
                created_after=None,
                created_before=None,
                due_date=None,
                health_status=None,
                issue_type=None,
                labels=None,
                scope=None,
                search=None,
                sort=None,
                state=None,
            ),
            "List issues in project 123",
        ),
        (
            ListIssuesInput(
                url="https://gitlab.com/namespace/project",
                assignee_id=None,
                assignee_usernames=None,
                author_id=None,
                author_username=None,
                confidential=None,
                created_after=None,
                created_before=None,
                due_date=None,
                health_status=None,
                issue_type=None,
                labels=None,
                scope=None,
                search=None,
                sort=None,
                state=None,
            ),
            "List issues in https://gitlab.com/namespace/project",
        ),
    ],
)
def test_list_issues_format_display_message(input_data, expected_message):
    tool = ListIssues(description="List issues description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_get_issue(issue_tool_setup):
    gitlab_client_mock, metadata = issue_tool_setup

    tool = GetIssue(description="get issue description", metadata=metadata)

    response = await tool._arun(project_id=1, issue_iid=123)

    expected_response = json.dumps(
        {
            "issue": {
                "id": 1,
                "title": "Test Issue",
                "description": "This is a test issue",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/issues/123", parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,expected_error",
    [
        # Missing project_id
        (
            {"issue_iid": 123},
            "'project_id' must be provided when 'url' is not",
        ),
        # Missing issue_iid
        (
            {"project_id": 1},
            "'issue_iid' must be provided when 'url' is not",
        ),
        # Missing both project_id and issue_iid
        (
            {},
            "'project_id' must be provided when 'url' is not; 'issue_iid' must be provided when 'url' is not",
        ),
    ],
)
async def test_get_issue_validation(kwargs, expected_error, issue_tool_setup):
    gitlab_client_mock, metadata = issue_tool_setup

    tool = GetIssue(description="get issue description", metadata=metadata)

    response = await tool._arun(**kwargs)
    response_json = json.loads(response)

    assert "error" in response_json
    assert expected_error in response_json["error"]
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,expected_path",
    URL_SUCCESS_CASES,
)
async def test_get_issue_with_url_success(
    url, project_id, issue_iid, expected_path, gitlab_client_mock, metadata, issue_data
):
    tool = GetIssue(description="get issue description", metadata=metadata)

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=issue_data,
    )

    expected_response = json.dumps(
        {
            "issue": {
                "id": 1,
                "title": "Test Issue",
                "description": "This is a test issue",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path, parse_json=False
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_get_issue_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = GetIssue(description="get issue description", metadata=metadata)

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,input_data,error_pattern",
    [
        # Tools requiring both project_id AND issue_iid
        (
            GetIssue,
            {"project_id": 1},
            "'issue_iid' must be provided when 'url' is not",
        ),
        (
            GetIssue,
            {"issue_iid": 123},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            GetIssue,
            {},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            UpdateIssue,
            {"project_id": 1},
            "'issue_iid' must be provided when 'url' is not",
        ),
        (
            UpdateIssue,
            {"issue_iid": 123},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            UpdateIssue,
            {},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            CreateIssueNote,
            {"project_id": 1, "body": "Test note"},
            "'issue_iid' must be provided when 'url' is not",
        ),
        (
            CreateIssueNote,
            {"issue_iid": 123, "body": "Test note"},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            CreateIssueNote,
            {"body": "Test note"},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            ListIssueNotes,
            {"project_id": 1},
            "'issue_iid' must be provided when 'url' is not",
        ),
        (
            ListIssueNotes,
            {"issue_iid": 123},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            ListIssueNotes,
            {},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            GetIssueNote,
            {"project_id": 1, "note_id": 1},
            "'issue_iid' must be provided when 'url' is not",
        ),
        (
            GetIssueNote,
            {"issue_iid": 123, "note_id": 1},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            GetIssueNote,
            {"note_id": 1},
            "'project_id' must be provided when 'url' is not",
        ),
        # Tools requiring only project_id
        (
            CreateIssue,
            {"title": "Test Issue"},
            "'project_id' must be provided when 'url' is not",
        ),
        (
            CreateIssue,
            {"url": None, "project_id": None, "title": "Test Issue"},
            "'project_id' must be provided when 'url' is not",
        ),
        (ListIssues, {}, "'project_id' must be provided when 'url' is not"),
        (
            ListIssues,
            {"url": None, "project_id": None},
            "'project_id' must be provided when 'url' is not",
        ),
    ],
)
async def test_tool_validation_error(metadata, tool_class, input_data, error_pattern):
    tool = tool_class(
        description=f"{tool_class.__name__.lower()} description", metadata=metadata
    )

    response = await tool._arun(**input_data)
    error_response = json.loads(response)

    assert "error" in error_response
    assert error_pattern in error_response["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_class,method_name,error_message,required_params",
    [
        (
            CreateIssue,
            "apost",
            "API error: Bad request",
            {"project_id": 1, "title": "Test Issue"},
        ),
        (
            GetIssue,
            "aget",
            "API error: Issue not found",
            {"project_id": 1, "issue_iid": 999},
        ),
        (
            UpdateIssue,
            "aput",
            "API error: Unauthorized",
            {"project_id": 1, "issue_iid": 123, "title": "Updated Test Issue"},
        ),
        (ListIssues, "aget", "API error: Server error", {"project_id": 1}),
        (
            CreateIssueNote,
            "apost",
            "API error: Note creation failed",
            {"project_id": 1, "issue_iid": 123, "body": "Test note"},
        ),
        (
            ListIssueNotes,
            "aget",
            "API error: Notes not accessible",
            {"project_id": 1, "issue_iid": 123},
        ),
        (
            GetIssueNote,
            "aget",
            "API error: Note not found",
            {"project_id": 1, "issue_iid": 123, "note_id": 999},
        ),
    ],
)
async def test_api_errors(
    tool_class,
    method_name,
    error_message,
    required_params,
    gitlab_client_mock,
    metadata,
):
    """Test API error handling for all tools."""
    # Set up the mock to raise an exception
    getattr(gitlab_client_mock, method_name).side_effect = Exception(error_message)

    tool = tool_class(
        description=f"{tool_class.__name__.lower()} description", metadata=metadata
    )

    response = await tool._arun(**required_params)

    error_response = json.loads(response)
    assert "error" in error_response
    assert error_message in error_response["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_data",
    [
        {"project_id": 1, "issue_iid": 123},  # Missing note_id
        {
            "url": "https://gitlab.com/namespace/project/-/issues/123"
        },  # Missing all required fields
    ],
)
async def test_get_issue_note_tool_validation_without_note_id(metadata, input_data):
    tool = GetIssueNote(description="get issue note description", metadata=metadata)

    with pytest.raises(ValueError, match="note_id\n  Field required"):
        await tool.arun(input_data)


@pytest.mark.asyncio
async def test_update_issue(gitlab_client_mock, metadata):
    gitlab_client_mock.aput = AsyncMock(
        return_value={
            "id": 123,
            "title": "Updated Test Issue",
            "description": "This is an updated test issue",
            "labels": ["bug", "critical"],
            "assignee_ids": [15, 16],
            "state": "closed",
        }
    )

    tool = UpdateIssue(description="update issue description", metadata=metadata)

    response = await tool._arun(
        project_id=1,
        issue_iid=123,
        title="Updated Test Issue",
        description="This is an updated test issue",
        labels="bug,critical",
        assignee_ids=[15, 16],
        state_event="close",
    )

    expected_response = json.dumps(
        {
            "updated_issue": {
                "id": 123,
                "title": "Updated Test Issue",
                "description": "This is an updated test issue",
                "labels": ["bug", "critical"],
                "assignee_ids": [15, 16],
                "state": "closed",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path="/api/v4/projects/1/issues/123",
        body=json.dumps(
            {
                "title": "Updated Test Issue",
                "description": "This is an updated test issue",
                "labels": "bug,critical",
                "assignee_ids": [15, 16],
                "state_event": "close",
            }
        ),
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            IssueResourceInput(project_id=123, issue_iid=456),
            "Read issue #456 in project 123",
        ),
        (
            IssueResourceInput(url="https://gitlab.com/namespace/project/-/issues/42"),
            "Read issue https://gitlab.com/namespace/project/-/issues/42",
        ),
    ],
)
def test_get_issue_format_display_message(input_data, expected_message):
    tool = GetIssue(description="Get issue description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            UpdateIssueInput(
                project_id=123,
                issue_iid=456,
                title=None,
                description=None,
                labels=None,
                assignee_ids=None,
                confidential=None,
                due_date=None,
                state_event=None,
                discussion_locked=None,
            ),
            "Update issue #456 in project 123",
        ),
        (
            UpdateIssueInput(
                url="https://gitlab.com/namespace/project/-/issues/42",
                title=None,
                description=None,
                labels=None,
                assignee_ids=None,
                confidential=None,
                due_date=None,
                state_event=None,
                discussion_locked=None,
            ),
            "Update issue https://gitlab.com/namespace/project/-/issues/42",
        ),
    ],
)
def test_update_issue_format_display_message(input_data, expected_message):
    tool = UpdateIssue(description="Update issue description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_create_issue_note(issue_tool_setup, note_data):
    gitlab_client_mock, metadata = issue_tool_setup
    gitlab_client_mock.apost.return_value = note_data

    tool = CreateIssueNote(
        description="create issue note description", metadata=metadata
    )

    response = await tool._arun(project_id=1, issue_iid=123, body="Test note")

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
        path="/api/v4/projects/1/issues/123/notes",
        body=json.dumps({"body": "Test note"}),
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateIssueNoteInput(
                project_id=123, issue_iid=456, body="This is a comment"
            ),
            "Add comment to issue #456 in project 123",
        ),
        (
            CreateIssueNoteInput(
                url="https://gitlab.com/namespace/project/-/issues/42",
                body="This is a comment",
            ),
            "Add comment to issue https://gitlab.com/namespace/project/-/issues/42",
        ),
    ],
)
def test_create_issue_note_format_display_message(input_data, expected_message):
    tool = CreateIssueNote(description="Create issue note description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sort,order_by,expected_params",
    [
        (None, None, {}),
        ("asc", None, {"sort": "asc"}),
        ("desc", "created_at", {"sort": "desc", "order_by": "created_at"}),
        (None, "updated_at", {"order_by": "updated_at"}),
    ],
)
async def test_list_issue_notes(
    sort, order_by, expected_params, gitlab_client_mock, metadata
):
    gitlab_client_mock.aget = AsyncMock(
        return_value=[
            {"id": 1, "body": "Note 1"},
            {"id": 2, "body": "Note 2"},
        ]
    )

    tool = ListIssueNotes(description="list issue notes description", metadata=metadata)

    response = await tool._arun(
        project_id=1, issue_iid=123, sort=sort, order_by=order_by
    )

    expected_response = json.dumps(
        {"notes": [{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/issues/123/notes",
        params=expected_params,
        parse_json=False,
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListIssueNotesInput(
                project_id=123, issue_iid=456, sort=None, order_by=None
            ),
            "Read comments on issue #456 in project 123",
        ),
        (
            ListIssueNotesInput(
                url="https://gitlab.com/namespace/project/-/issues/42",
                sort="asc",
                order_by="created_at",
            ),
            "Read comments on issue https://gitlab.com/namespace/project/-/issues/42",
        ),
    ],
)
def test_list_issue_notes_format_display_message(input_data, expected_message):
    tool = ListIssueNotes(description="List issue notes description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_get_issue_note(issue_tool_setup, note_data):
    gitlab_client_mock, metadata = issue_tool_setup
    gitlab_client_mock.aget.return_value = note_data

    tool = GetIssueNote(description="get issue note description", metadata=metadata)

    response = await tool._arun(project_id=1, issue_iid=123, note_id=1)

    expected_response = json.dumps(
        {
            "note": {
                "id": 1,
                "body": "Test note",
                "created_at": "2024-01-01T12:00:00Z",
                "author": {"id": 1, "name": "Test User"},
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1/issues/123/notes/1", parse_json=False
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            GetIssueNoteInput(project_id=123, issue_iid=456, note_id=789),
            "Read comment #789 on issue #456 in project 123",
        ),
        (
            GetIssueNoteInput(
                url="https://gitlab.com/namespace/project/-/issues/42", note_id=789
            ),
            "Read comment #789 on issue https://gitlab.com/namespace/project/-/issues/42",
        ),
    ],
)
def test_get_issue_note_format_display_message(input_data, expected_message):
    tool = GetIssueNote(description="Get issue note description")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,expected_path",
    URL_SUCCESS_CASES,
)
async def test_update_issue_with_url_success(
    url, project_id, issue_iid, expected_path, gitlab_client_mock, metadata
):
    update_data = {
        "id": 123,
        "title": "Updated Test Issue",
        "description": "This is an updated test issue",
    }

    tool = UpdateIssue(description="update issue description", metadata=metadata)

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=update_data,
        title="Updated Test Issue",
        description="This is an updated test issue",
    )

    expected_response = json.dumps(
        {
            "updated_issue": {
                "id": 123,
                "title": "Updated Test Issue",
                "description": "This is an updated test issue",
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aput.assert_called_once_with(
        path=expected_path,
        body=json.dumps(
            {
                "title": "Updated Test Issue",
                "description": "This is an updated test issue",
            }
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_update_issue_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = UpdateIssue(description="update issue description", metadata=metadata)

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        title="Updated Test Issue",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,expected_path",
    [
        # Modify paths for notes endpoints
        (
            "https://gitlab.com/namespace/project/-/issues/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/issues/123/notes",
        ),
        (
            "https://gitlab.com/namespace/project/-/issues/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/issues/123/notes",
        ),
    ],
)
async def test_create_issue_note_with_url_success(
    url, project_id, issue_iid, expected_path, gitlab_client_mock, metadata, note_data
):
    tool = CreateIssueNote(
        description="create issue note description", metadata=metadata
    )

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
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
    "url,project_id,issue_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_create_issue_note_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = CreateIssueNote(
        description="create issue note description", metadata=metadata
    )

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        body="Test note",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,expected_path",
    [
        # Modify paths for notes endpoints
        (
            "https://gitlab.com/namespace/project/-/issues/123",
            None,
            None,
            "/api/v4/projects/namespace%2Fproject/issues/123/notes",
        ),
        (
            "https://gitlab.com/namespace/project/-/issues/123",
            "namespace%2Fproject",
            123,
            "/api/v4/projects/namespace%2Fproject/issues/123/notes",
        ),
    ],
)
async def test_list_issue_notes_with_url_success(
    url, project_id, issue_iid, expected_path, gitlab_client_mock, metadata
):
    notes_data = [
        {"id": 1, "body": "Note 1"},
        {"id": 2, "body": "Note 2"},
    ]

    tool = ListIssueNotes(description="list issue notes description", metadata=metadata)

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=notes_data,
        sort="asc",
    )

    expected_response = json.dumps(
        {"notes": [{"id": 1, "body": "Note 1"}, {"id": 2, "body": "Note 2"}]}
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        params={"sort": "asc"},
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_list_issue_notes_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = ListIssueNotes(description="list issue notes description", metadata=metadata)

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,note_id,expected_path",
    [
        # Modify paths for notes endpoints with note_id
        (
            "https://gitlab.com/namespace/project/-/issues/123",
            None,
            None,
            1,
            "/api/v4/projects/namespace%2Fproject/issues/123/notes/1",
        ),
        (
            "https://gitlab.com/namespace/project/-/issues/123",
            "namespace%2Fproject",
            123,
            1,
            "/api/v4/projects/namespace%2Fproject/issues/123/notes/1",
        ),
    ],
)
async def test_get_issue_note_with_url_success(
    url,
    project_id,
    issue_iid,
    note_id,
    expected_path,
    gitlab_client_mock,
    metadata,
    note_data,
):
    tool = GetIssueNote(description="get issue note description", metadata=metadata)

    response = await tool_url_success_response(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        gitlab_client_mock=gitlab_client_mock,
        response_data=note_data,
        note_id=note_id,
    )

    expected_response = json.dumps(
        {
            "note": {
                "id": 1,
                "body": "Test note",
                "created_at": "2024-01-01T12:00:00Z",
                "author": {"id": 1, "name": "Test User"},
            }
        }
    )
    assert response == expected_response

    gitlab_client_mock.aget.assert_called_once_with(
        path=expected_path,
        parse_json=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,project_id,issue_iid,error_contains",
    URL_ERROR_CASES,
)
async def test_get_issue_note_with_url_error(
    url, project_id, issue_iid, error_contains, gitlab_client_mock, metadata
):
    tool = GetIssueNote(description="get issue note description", metadata=metadata)

    await assert_tool_url_error(
        tool=tool,
        url=url,
        project_id=project_id,
        issue_iid=issue_iid,
        error_contains=error_contains,
        gitlab_client_mock=gitlab_client_mock,
        note_id=1,
    )
