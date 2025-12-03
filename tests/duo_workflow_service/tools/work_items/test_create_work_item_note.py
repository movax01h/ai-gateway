# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.work_item import (
    CreateWorkItemNote,
    CreateWorkItemNoteInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    mock.graphql = AsyncMock()
    return mock


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.fixture(name="work_item_data")
def work_item_data_fixture():
    """Fixture for sample work item data."""
    return {
        "id": "gid://gitlab/WorkItem/123",
        "iid": "42",
        "title": "Test Work Item",
        "description": "This is a test work item",
        "state": "opened",
        "createdAt": "2025-04-29T11:35:36.000+02:00",
        "updatedAt": "2025-04-29T12:35:36.000+02:00",
        "author": {"username": "test_user", "name": "Test User"},
        "workItemType": {"name": "Issue"},
    }


@pytest.fixture(name="created_note_data_fixture")
def created_note_data_fixture_func():
    """Fixture for created note data."""
    return {
        "id": "gid://gitlab/Note/456",
        "body": "This is a test comment",
        "createdAt": "2025-04-29T13:35:36.000+02:00",
        "author": {"username": "test_user", "name": "Test User"},
    }


@pytest.mark.parametrize(
    "params,response_key,expected_body",
    [
        (
            {
                "group_id": "namespace/group",
                "work_item_iid": 42,
                "body": "This is a test comment",
            },
            "namespace",
            "This is a test comment",
        ),
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This is a test comment",
            },
            "project",
            "This is a test comment",
        ),
        (
            {
                "url": "https://gitlab.com/groups/namespace/group/-/work_items/42",
                "body": "This is a test comment via URL",
            },
            "namespace",
            "This is a test comment via URL",
        ),
        (
            {
                "url": "https://gitlab.com/namespace/project/-/work_items/42",
                "body": "This is a test comment via project URL",
            },
            "project",
            "This is a test comment via project URL",
        ),
    ],
)
@pytest.mark.asyncio
async def test_create_work_item_note_success(
    gitlab_client_mock,
    metadata,
    work_item_data,
    created_note_data_fixture,
    params,
    response_key,
    expected_body,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        {response_key: {"workItems": {"nodes": [work_item_data]}}},
        {"createNote": {"note": created_note_data_fixture, "errors": []}},
    ]

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)
    response = await tool._arun(**params)

    response_json = json.loads(response)
    assert response_json["status"] == "success"
    assert "message" in response_json
    assert "Note created successfully" in response_json["message"]
    assert response_json["note"] == created_note_data_fixture

    # Verify GraphQL was called twice
    assert gitlab_client_mock.graphql.call_count == 2

    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "createNote" in second_call_args[0]
    assert second_call_args[1]["input"]["noteableId"] == work_item_data["id"]
    assert second_call_args[1]["input"]["body"] == expected_body


@pytest.mark.asyncio
async def test_create_work_item_note_with_optional_parameters(
    gitlab_client_mock, metadata, work_item_data, created_note_data_fixture
):
    # Mock the GraphQL calls
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        {"project": {"workItems": {"nodes": [work_item_data]}}},
        {"createNote": {"note": created_note_data_fixture, "errors": []}},
    ]

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        body="This is an internal comment",
        internal=True,
        discussion_id="gid://gitlab/Discussion/789",
    )

    response_json = json.loads(response)
    assert response_json["status"] == "success"

    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    note_input = second_call_args[1]["input"]
    assert note_input["body"] == "This is an internal comment"
    assert note_input["internal"] is True
    assert note_input["discussionId"] == "gid://gitlab/Discussion/789"


@pytest.mark.parametrize(
    "params,graphql_responses,expected_error,expected_details,expected_call_count",
    [
        # Work item not found
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 999,
                "body": "This comment won't be created",
            },
            [{"project": {"workItems": {"nodes": []}}}],
            "Work item not found",
            None,
            1,
        ),
        # Missing work item ID
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This comment won't be created",
            },
            [
                {
                    "project": {
                        "workItems": {
                            "nodes": [{"iid": "42", "title": "Test Work Item"}]
                        }
                    }
                }
            ],
            "Work item exists but has no ID field",
            None,
            1,
        ),
        # Create note error
        (
            {"project_id": "namespace/project", "work_item_iid": 42, "body": ""},
            [
                {
                    "project": {
                        "workItems": {
                            "nodes": [
                                {
                                    "id": "gid://gitlab/WorkItem/123",
                                    "iid": "42",
                                    "title": "Test Work Item",
                                }
                            ]
                        }
                    }
                },
                {"createNote": {"note": None, "errors": ["Body cannot be blank"]}},
            ],
            "Failed to create note",
            {"graphql_errors": None, "note_errors": ["Body cannot be blank"]},
            2,
        ),
        # GraphQL error on create
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This comment will fail",
            },
            [
                {
                    "project": {
                        "workItems": {
                            "nodes": [
                                {
                                    "id": "gid://gitlab/WorkItem/123",
                                    "iid": "42",
                                    "title": "Test Work Item",
                                }
                            ]
                        }
                    }
                },
                {"errors": ["GraphQL syntax error"]},
            ],
            "GraphQL syntax error",
            None,
            2,
        ),
        # GraphQL error on fetch
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This comment will fail",
            },
            [Exception("GraphQL connection error")],
            "GraphQL connection error",
            None,
            1,
        ),
        # Invalid URL
        (
            {
                "url": "https://gitlab.com/invalid-url",
                "body": "This comment won't be created",
            },
            [],
            "Failed to parse work item URL: Not a work item URL: https://gitlab.com/invalid-url",
            None,
            0,
        ),
        # Missing IID
        (
            {
                "project_id": "namespace/project",
                "body": "This comment won't be created",
            },
            [],
            "Must provide work_item_iid if no URL is given",
            None,
            0,
        ),
        # Missing root key
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This comment won't be created",
            },
            [{}],
            "No project found in response",
            None,
            1,
        ),
    ],
)
@pytest.mark.asyncio
async def test_create_work_item_note_errors(
    gitlab_client_mock,
    metadata,
    params,
    graphql_responses,
    expected_error,
    expected_details,
    expected_call_count,
):
    if graphql_responses and isinstance(graphql_responses[0], Exception):
        gitlab_client_mock.graphql = AsyncMock(side_effect=graphql_responses[0])
    elif graphql_responses:
        gitlab_client_mock.graphql = AsyncMock(side_effect=graphql_responses)

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)
    response = await tool._arun(**params)
    response_json = json.loads(response)

    assert "error" in response_json
    assert expected_error in response_json["error"]

    if expected_details:
        assert "details" in response_json
        assert response_json["details"] == expected_details

    assert gitlab_client_mock.graphql.call_count == expected_call_count


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateWorkItemNoteInput(
                group_id="namespace/group", work_item_iid=42, body="Test comment"
            ),
            "Add comment to work item #42 in group namespace/group",
        ),
        (
            CreateWorkItemNoteInput(
                project_id="namespace/project", work_item_iid=42, body="Test comment"
            ),
            "Add comment to work item #42 in project namespace/project",
        ),
        (
            CreateWorkItemNoteInput(
                url="https://gitlab.com/namespace/project/-/work_items/42",
                body="Test comment",
            ),
            "Add comment to work item https://gitlab.com/namespace/project/-/work_items/42",
        ),
    ],
)
def test_create_work_item_note_format_display_message(input_data, expected_message):
    tool = CreateWorkItemNote(description="create work item note")
    message = tool.format_display_message(input_data)
    assert message == expected_message
