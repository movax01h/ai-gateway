# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.executor_http_client import (
    RESOURCE_NOT_AVAILABLE_MESSAGE,
    GitLabGraphQLError,
)
from duo_workflow_service.tools.work_item import (
    CreateWorkItemNote,
    CreateWorkItemNoteInput,
)
from lib.context import gitlab_version


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


@pytest.fixture(name="report_gitlab_version")
def report_gitlab_version_fixture():
    """Set the version both the gate and the warning message read.

    Defaults to one whose createDiscussion takes an ai_workflows token.
    """
    token = gitlab_version.set("19.4.0")

    yield gitlab_version.set

    gitlab_version.reset(token)


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
    )

    response_json = json.loads(response)
    assert response_json["status"] == "success"

    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    note_input = second_call_args[1]["input"]
    assert note_input["body"] == "This is an internal comment"
    assert note_input["internal"] is True


@pytest.mark.asyncio
async def test_create_work_item_note_reply_to_note(
    gitlab_client_mock, metadata, work_item_data, created_note_data_fixture
):
    """Test creating a reply to an existing note using note_id."""
    # Mock the GraphQL calls
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        {"project": {"workItems": {"nodes": [work_item_data]}}},
        {
            "note": {
                "id": "gid://gitlab/Note/456",
                "discussion": {"replyId": "gid://gitlab/Discussion/789"},
            }
        },
        {"createNote": {"note": created_note_data_fixture, "errors": []}},
    ]

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        body="This is a reply to a comment",
        note_id=456,
    )

    response_json = json.loads(response)
    assert response_json["status"] == "success"

    assert gitlab_client_mock.graphql.call_count == 3

    third_call_args = gitlab_client_mock.graphql.call_args_list[2][0]
    note_input = third_call_args[1]["input"]
    assert note_input["body"] == "This is a reply to a comment"
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
        # Create discussion error
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "",
                "start_discussion": True,
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
                {
                    "createDiscussion": {
                        "note": None,
                        "errors": ["Body cannot be blank"],
                    }
                },
            ],
            "Failed to create discussion",
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
            "Project 'namespace/project' not found or not accessible",
            None,
            1,
        ),
        # Note not found when using note_id
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This reply will fail",
                "note_id": 999,
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
                {"note": None},
            ],
            "No note found for ID 999",
            None,
            2,
        ),
        # Note has no discussion when using note_id
        (
            {
                "project_id": "namespace/project",
                "work_item_iid": 42,
                "body": "This reply will fail",
                "note_id": 456,
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
                {
                    "note": {
                        "id": "gid://gitlab/Note/456",
                        "discussion": None,
                    }
                },
            ],
            "Note 456 exists but is not part of a discussion",
            None,
            2,
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.usefixtures("report_gitlab_version")
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

    # Raw exceptions (like connection errors) are not wrapped in ToolException
    if graphql_responses and isinstance(graphql_responses[0], Exception):
        with pytest.raises(Exception, match=expected_error):
            await tool._arun(**params)
    else:
        with pytest.raises(ToolException) as exc_info:
            await tool._arun(**params)

        exception_message = str(exc_info.value)
        assert expected_error in exception_message

        # Verify details are embedded in the exception message
        if expected_details:
            if expected_details.get("note_errors"):
                for error in expected_details["note_errors"]:
                    assert error in exception_message

    assert gitlab_client_mock.graphql.call_count == expected_call_count


@pytest.mark.asyncio
@pytest.mark.usefixtures("report_gitlab_version")
async def test_create_work_item_note_start_discussion(
    gitlab_client_mock, metadata, work_item_data, created_note_data_fixture
):
    """start_discussion=True calls createDiscussion instead of createNote."""
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=[
            {"project": {"workItems": {"nodes": [work_item_data]}}},
            {"createDiscussion": {"note": created_note_data_fixture, "errors": []}},
        ]
    )

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        body="This is a question",
        start_discussion=True,
    )

    response_json = json.loads(response)
    assert response_json["status"] == "success"
    assert response_json["message"] == "Discussion created successfully."
    assert response_json["note"] == created_note_data_fixture
    assert "warnings" not in response_json

    assert gitlab_client_mock.graphql.call_count == 2
    mutation, variables = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "createDiscussion" in mutation
    assert "createNote" not in mutation
    assert variables["input"]["body"] == "This is a question"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version,expected_version_phrase",
    [
        ("19.3.0", "this instance reports 19.3.0."),
        # An instance that reported nothing must not be quoted a version it
        # never sent - get_gitlab_version() would substitute its 18.6.0 floor.
        (None, "this instance reports no version."),
    ],
)
async def test_create_work_item_note_start_discussion_falls_back_below_19_4(
    gitlab_client_mock,
    metadata,
    work_item_data,
    created_note_data_fixture,
    report_gitlab_version,
    version,
    expected_version_phrase,
):
    """Below 19.4 the mutation rejects an ai_workflows token, so post a plain note."""
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=[
            {"project": {"workItems": {"nodes": [work_item_data]}}},
            {"createNote": {"note": created_note_data_fixture, "errors": []}},
        ]
    )
    report_gitlab_version(version)

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        body="This is a question",
        start_discussion=True,
    )

    response_json = json.loads(response)
    assert response_json["status"] == "success"
    assert response_json["message"] == "Note created successfully."
    warning = response_json["warnings"][0]
    assert "GitLab 19.4 or later" in warning
    assert expected_version_phrase in warning

    mutation, _ = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "createNote" in mutation
    assert "createDiscussion" not in mutation


@pytest.mark.asyncio
@pytest.mark.usefixtures("report_gitlab_version")
async def test_create_work_item_note_start_discussion_falls_back_when_rejected(
    gitlab_client_mock,
    metadata,
    work_item_data,
    created_note_data_fixture,
):
    """On 19.4+ a rejected createDiscussion still posts the comment via createNote."""
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=[
            {"project": {"workItems": {"nodes": [work_item_data]}}},
            GitLabGraphQLError([{"message": RESOURCE_NOT_AVAILABLE_MESSAGE}]),
            {"createNote": {"note": created_note_data_fixture, "errors": []}},
        ]
    )

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        body="This is a question",
        start_discussion=True,
    )

    response_json = json.loads(response)
    assert response_json["status"] == "success"
    assert response_json["message"] == "Note created successfully."
    assert "rejected createDiscussion" in response_json["warnings"][0]

    mutations = [call[0][0] for call in gitlab_client_mock.graphql.call_args_list]
    assert "createDiscussion" in mutations[1]
    assert "createNote" in mutations[2]


@pytest.mark.asyncio
@pytest.mark.usefixtures("report_gitlab_version")
@pytest.mark.parametrize(
    "errors",
    [
        # Raised after the discussion was saved, so a createNote would duplicate it.
        [{"message": "Internal server error"}],
        [
            {"message": RESOURCE_NOT_AVAILABLE_MESSAGE},
            {"message": "Internal server error"},
        ],
        [],
    ],
)
async def test_create_work_item_note_start_discussion_other_graphql_errors_do_not_fall_back(
    gitlab_client_mock, metadata, work_item_data, errors
):
    """Only authorization rejections are safe to retry as a plain note."""
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=[
            {"project": {"workItems": {"nodes": [work_item_data]}}},
            GitLabGraphQLError(errors),
        ]
    )

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    with pytest.raises(GitLabGraphQLError):
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            body="This is a question",
            start_discussion=True,
        )

    assert gitlab_client_mock.graphql.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("report_gitlab_version")
async def test_create_work_item_note_start_discussion_timeout_does_not_fall_back(
    gitlab_client_mock, metadata, work_item_data
):
    """A timeout may have created the discussion, so a createNote retry could post twice."""
    gitlab_client_mock.graphql = AsyncMock(
        side_effect=[
            {"project": {"workItems": {"nodes": [work_item_data]}}},
            Exception("GraphQL request timed out after 10.0 seconds"),
        ]
    )

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    with pytest.raises(Exception, match="timed out"):
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            body="This is a question",
            start_discussion=True,
        )

    assert gitlab_client_mock.graphql.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures("report_gitlab_version")
async def test_create_work_item_note_start_discussion_with_note_id_raises(
    gitlab_client_mock, metadata
):
    """start_discussion cannot be combined with note_id, and costs no round-trip."""
    gitlab_client_mock.graphql = AsyncMock()

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    with pytest.raises(ToolException, match="start_discussion cannot be combined"):
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            body="This is a reply",
            note_id=456,
            start_discussion=True,
        )

    assert gitlab_client_mock.graphql.call_count == 0


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
