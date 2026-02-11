# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

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


@pytest.fixture(name="version_variables")
def version_variables_default_fixture():
    """Fixture for note-specific version variables."""
    return {
        "includeNoteResolvedAndResolvableFields": True,
        "includeDiscussionIdField": True,
    }


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_group_id(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = {
        "namespace": {
            "workItems": {
                "nodes": [{"widgets": [{"notes": {"nodes": work_item_notes}}]}]
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(group_id="namespace/group", work_item_iid=42)

    expected_response = json.dumps({"notes": work_item_notes}, indent=2)
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()

    # Verify version-specific variables are passed to GraphQL query
    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["fullPath"] == "namespace/group"
    assert query_variables["workItemIid"] == "42"
    assert query_variables["includeNoteResolvedAndResolvableFields"] is True
    assert query_variables["includeDiscussionIdField"] is True


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_calls_version_compatibility(
    mock_get_query_variables,
    metadata,
):
    mock_get_query_variables.return_value = {
        "includeNoteResolvedAndResolvableFields": False,
        "includeDiscussionIdField": True,
    }

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
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = {
        "project": {
            "workItems": {
                "nodes": [{"widgets": [{"notes": {"nodes": work_item_notes}}]}]
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"notes": work_item_notes}, indent=2)
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()

    # Verify version-specific variables are passed to GraphQL query
    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["fullPath"] == "namespace/project"
    assert query_variables["workItemIid"] == "42"
    assert query_variables["includeNoteResolvedAndResolvableFields"] is True
    assert query_variables["includeDiscussionIdField"] is True


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_get_work_item_notes_with_group_url(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
    work_item_notes,
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    # Mock the _validate_work_item_url method
    resolved_work_item = ResolvedWorkItem(
        parent=ResolvedParent(type="group", full_path="namespace/group"),
        work_item_iid=42,
    )
    graphql_response = {
        "namespace": {
            "workItems": {
                "nodes": [{"widgets": [{"notes": {"nodes": work_item_notes}}]}]
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)
    tool._validate_work_item_url = AsyncMock(return_value=resolved_work_item)

    response = await tool._arun(
        url="https://gitlab.com/groups/namespace/group/-/work_items/42"
    )

    expected_response = json.dumps({"notes": work_item_notes}, indent=2)
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
    version_variables,
):
    mock_get_query_variables.return_value = version_variables
    graphql_response = {
        "project": {
            "workItems": {
                "nodes": [{"widgets": [{"notes": {"nodes": work_item_notes}}]}]
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/work_items/42"
    )

    expected_response = json.dumps({"notes": work_item_notes}, indent=2)
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

    expected_response = json.dumps({"notes": []})
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
    graphql_response = {
        "project": {"workItems": {"nodes": [{"widgets": [{"notes": {"nodes": []}}]}]}}
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"notes": []}, indent=2)
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

    response = await tool._arun(project_id="namespace/project", work_item_iid=999)

    expected_response = json.dumps({"error": "No work item found."})
    assert response == expected_response

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

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"error": "GraphQL error"})
    assert response == expected_response

    mock_get_query_variables.assert_called_once_with(
        "includeNoteResolvedAndResolvableFields", "includeDiscussionIdField"
    )
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_invalid_url(gitlab_client_mock, metadata):
    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(url="https://gitlab.com/invalid-url")

    response_json = json.loads(response)
    assert "error" in response_json
    assert (
        "Failed to parse work item URL: Not a work item URL: https://gitlab.com/invalid-url"
        in response_json["error"]
    )
    gitlab_client_mock.graphql.assert_not_called()


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
