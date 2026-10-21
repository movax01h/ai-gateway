# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.work_item import GetWorkItemNotes, GetWorkItemNotesInput
from duo_workflow_service.tools.work_items.base_tool import (
    ResolvedParent,
    ResolvedWorkItem,
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


@pytest.fixture(name="work_item_notes")
def work_item_notes_fixture():
    """Fixture for sample work item notes."""
    return [
        {
            "id": "gid://gitlab/Note/123",
            "body": "This is the first comment",
            "bodyHtml": "<p>This is the first comment</p>",
            "createdAt": "2025-04-29T11:35:36.000+02:00",
            "updatedAt": "2025-04-29T11:35:36.000+02:00",
            "author": {"username": "test_user", "name": "Test User"},
        },
        {
            "id": "gid://gitlab/Note/124",
            "body": "This is a reply to the first comment",
            "bodyHtml": "<p>This is a reply to the first comment</p>",
            "createdAt": "2025-04-29T12:35:36.000+02:00",
            "updatedAt": "2025-04-29T12:35:36.000+02:00",
            "author": {"username": "another_user", "name": "Another User"},
        },
    ]


@pytest.fixture(name="page_info")
def page_info_fixture():
    """Fixture for a default pageInfo with no next page."""
    return {"hasNextPage": False, "endCursor": None}


@pytest.fixture(name="version_variables")
def version_variables_default_fixture():
    """Fixture for note-specific version variables."""
    return {
        "includeNoteResolvedAndResolvableFields": True,
        "includeDiscussionIdField": True,
    }


def make_graphql_response(root_key, notes, page_info):
    """Helper to build a GraphQL response with notes and pageInfo."""
    return {
        root_key: {
            "workItems": {
                "nodes": [
                    {
                        "widgets": [
                            {
                                "notes": {
                                    "nodes": notes,
                                    "pageInfo": page_info,
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_group_id(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    page_info,
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = make_graphql_response("namespace", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(group_id="namespace/group", work_item_iid=42)

    expected_response = json.dumps(
        {"notes": work_item_notes, "page_info": page_info}, indent=2
    )
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()

    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["fullPath"] == "namespace/group"
    assert query_variables["workItemIid"] == "42"
    assert query_variables["first"] == 20
    assert query_variables["after"] is None
    assert query_variables["includeNoteResolvedAndResolvableFields"] is True
    assert query_variables["includeDiscussionIdField"] is True


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_calls_version_compatibility(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
):
    version_vars = {
        "includeNoteResolvedAndResolvableFields": False,
        "includeDiscussionIdField": True,
    }
    mock_get_query_variables.return_value = version_vars
    graphql_response = {
        "namespace": {
            "workItems": {
                "nodes": [{"widgets": [{"notes": {"nodes": work_item_notes}}]}]
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    await tool._arun(group_id="namespace/group", work_item_iid=42)

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_project_id(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    page_info,
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = make_graphql_response("project", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps(
        {"notes": work_item_notes, "page_info": page_info}, indent=2
    )
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()

    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["fullPath"] == "namespace/project"
    assert query_variables["workItemIid"] == "42"
    assert query_variables["first"] == 20
    assert query_variables["after"] is None
    assert query_variables["includeNoteResolvedAndResolvableFields"] is True
    assert query_variables["includeDiscussionIdField"] is True


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_group_url(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    page_info,
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    resolved_work_item = ResolvedWorkItem(
        parent=ResolvedParent(type="group", full_path="namespace/group"),
        work_item_iid=42,
    )
    graphql_response = make_graphql_response("namespace", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)
    tool._validate_work_item_url = AsyncMock(return_value=resolved_work_item)

    response = await tool._arun(
        url="https://gitlab.com/groups/namespace/group/-/work_items/42"
    )

    expected_response = json.dumps(
        {"notes": work_item_notes, "page_info": page_info}, indent=2
    )
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_project_url(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    page_info,
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = make_graphql_response("project", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/work_items/42"
    )

    expected_response = json.dumps(
        {"notes": work_item_notes, "page_info": page_info}, indent=2
    )
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_no_widgets(
    mock_get_query_variables, gitlab_client_mock, metadata, version_variables
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = {"project": {"workItems": {"nodes": [{"widgets": []}]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"notes": [], "page_info": {}}, indent=2)
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_empty_notes(
    mock_get_query_variables, gitlab_client_mock, metadata, version_variables
):
    mock_get_query_variables.return_value = version_variables
    page_info = {"hasNextPage": False, "endCursor": None}
    graphql_response = {
        "project": {
            "workItems": {
                "nodes": [
                    {"widgets": [{"notes": {"nodes": [], "pageInfo": page_info}}]}
                ]
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"notes": [], "page_info": page_info}, indent=2)
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_not_found(
    mock_get_query_variables, gitlab_client_mock, metadata, version_variables
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = {"project": {"workItems": {"nodes": []}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_id="namespace/project", work_item_iid=999)
    assert "No work item found." in str(exc_info.value)

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_graphql_error(
    mock_get_query_variables, gitlab_client_mock, metadata, version_variables
):
    mock_get_query_variables.return_value = version_variables
    gitlab_client_mock.graphql = AsyncMock(side_effect=Exception("GraphQL error"))

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    with pytest.raises(Exception, match="GraphQL error"):
        await tool._arun(project_id="namespace/project", work_item_iid=42)

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_invalid_url(gitlab_client_mock, metadata):
    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(url="https://gitlab.com/invalid-url")
    assert (
        "Failed to parse work item URL: Not a work item URL: https://gitlab.com/invalid-url"
        in str(exc_info.value)
    )
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_pagination_first_page(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    version_variables,
):
    """Test that first page returns notes with hasNextPage=True and a cursor."""
    mock_get_query_variables.return_value = version_variables
    page_info = {"hasNextPage": True, "endCursor": "cursor_abc123"}
    graphql_response = make_graphql_response("project", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project", work_item_iid=42, page_size=2
    )

    response_json = json.loads(response)
    assert response_json["notes"] == work_item_notes
    assert response_json["page_info"]["hasNextPage"] is True
    assert response_json["page_info"]["endCursor"] == "cursor_abc123"

    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["first"] == 2
    assert query_variables["after"] is None


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_pagination_subsequent_page(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    version_variables,
):
    """Test that subsequent page passes the cursor and returns the next batch."""
    mock_get_query_variables.return_value = version_variables
    page_info = {"hasNextPage": False, "endCursor": None}
    graphql_response = make_graphql_response("project", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        page_size=2,
        pagination_cursor="cursor_abc123",
    )

    response_json = json.loads(response)
    assert response_json["notes"] == work_item_notes
    assert response_json["page_info"]["hasNextPage"] is False

    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["first"] == 2
    assert query_variables["after"] == "cursor_abc123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_kwargs,expected_filter",
    [
        ({}, "ALL_NOTES"),
        ({"notes_filter": "ALL_NOTES"}, "ALL_NOTES"),
        ({"notes_filter": "ONLY_COMMENTS"}, "ONLY_COMMENTS"),
        ({"notes_filter": "ONLY_ACTIVITY"}, "ONLY_ACTIVITY"),
    ],
)
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_filter(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    page_info,
    version_variables,
    input_kwargs,
    expected_filter,
):
    """Test that notes_filter defaults to ALL_NOTES and is passed through as `filter`."""
    mock_get_query_variables.return_value = version_variables
    graphql_response = make_graphql_response("project", work_item_notes, page_info)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    await tool._arun(project_id="namespace/project", work_item_iid=42, **input_kwargs)

    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["filter"] == expected_filter


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            GetWorkItemNotesInput(group_id="namespace/group", work_item_iid=42),
            "Read comments on work item #42 in group namespace/group",
        ),
        (
            GetWorkItemNotesInput(project_id="namespace/project", work_item_iid=42),
            "Read comments on work item #42 in project namespace/project",
        ),
        (
            GetWorkItemNotesInput(
                url="https://gitlab.com/namespace/project/-/work_items/42"
            ),
            "Read comments on work item https://gitlab.com/namespace/project/-/work_items/42",
        ),
    ],
)
def test_get_work_item_notes_format_display_message(input_data, expected_message):
    tool = GetWorkItemNotes(description="get work item notes")
    message = tool.format_display_message(input_data)
    assert message == expected_message
