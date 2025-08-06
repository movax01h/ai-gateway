import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.work_item import (
    CreateWorkItem,
    CreateWorkItemInput,
    GetWorkItem,
    GetWorkItemNotes,
    GetWorkItemNotesInput,
    ListWorkItems,
    ListWorkItemsInput,
    ResolvedParent,
    ResolvedWorkItem,
    WorkItemResourceInput,
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
    }


@pytest.fixture(name="work_items_list")
def work_items_list_fixture():
    """Fixture for a list of work items."""
    return [
        {
            "id": "gid://gitlab/WorkItem/123",
            "iid": "42",
            "title": "Test Work Item 1",
            "state": "opened",
            "createdAt": "2025-04-29T11:35:36.000+02:00",
            "updatedAt": "2025-04-29T12:35:36.000+02:00",
            "author": {"username": "test_user", "name": "Test User"},
        },
        {
            "id": "gid://gitlab/WorkItem/124",
            "iid": "43",
            "title": "Test Work Item 2",
            "state": "closed",
            "createdAt": "2025-04-28T11:35:36.000+02:00",
            "updatedAt": "2025-04-28T12:35:36.000+02:00",
            "author": {"username": "test_user", "name": "Test User"},
        },
    ]


@pytest.mark.asyncio
async def test_validate_parent_url_with_group_id(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url=None,
        group_id="namespace/group",
        project_id=None,
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "group"
    assert result.full_path == "namespace/group"


@pytest.mark.asyncio
async def test_validate_parent_url_with_project_id(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url=None,
        group_id=None,
        project_id="namespace/project",
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "project"
    assert result.full_path == "namespace/project"


@pytest.mark.asyncio
async def test_validate_parent_url_with_group_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url="https://gitlab.com/groups/namespace/group",
        group_id=None,
        project_id=None,
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "group"
    assert result.full_path == "namespace/group"


@pytest.mark.asyncio
async def test_validate_parent_url_with_project_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url="https://gitlab.com/namespace/project",
        group_id=None,
        project_id=None,
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "project"
    assert result.full_path == "namespace/project"


@pytest.mark.asyncio
async def test_validate_parent_url_with_invalid_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url="https://example.com/not-gitlab",
        group_id=None,
        project_id=None,
    )
    assert isinstance(result, str)
    assert "Failed to parse parent work item URL" in result


@pytest.mark.asyncio
async def test_validate_parent_url_with_no_params(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(url=None, group_id=None, project_id=None)
    assert isinstance(result, str)
    assert "Must provide either URL, group_id, or project_id" in result


@pytest.mark.asyncio
async def test_validate_work_item_url_with_group_id_and_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    resolved_parent = ResolvedParent(type="group", full_path="namespace/group")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    result = await tool._validate_work_item_url(
        url=None,
        group_id="namespace/group",
        project_id=None,
        work_item_iid=42,
    )
    assert isinstance(result, ResolvedWorkItem)
    assert result.parent.type == "group"
    assert result.parent.full_path == "namespace/group"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_project_id_and_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    resolved_parent = ResolvedParent(type="project", full_path="namespace/project")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    result = await tool._validate_work_item_url(
        url=None,
        group_id=None,
        project_id="namespace/project",
        work_item_iid=42,
    )
    assert isinstance(result, ResolvedWorkItem)
    assert result.parent.type == "project"
    assert result.parent.full_path == "namespace/project"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_group_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_work_item_url(
        url="https://gitlab.com/groups/namespace/group/-/work_items/42",
        group_id=None,
        project_id=None,
        work_item_iid=None,
    )
    assert isinstance(result, ResolvedWorkItem)
    assert result.parent.type == "group"
    assert result.parent.full_path == "namespace/group"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_project_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_work_item_url(
        url="https://gitlab.com/namespace/project/-/work_items/42",
        group_id=None,
        project_id=None,
        work_item_iid=None,
    )
    assert isinstance(result, ResolvedWorkItem)
    assert result.parent.type == "project"
    assert result.parent.full_path == "namespace/project"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_no_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_work_item_url(
        url=None,
        group_id="namespace/group",
        project_id=None,
        work_item_iid=None,
    )
    assert isinstance(result, str)
    assert "Must provide work_item_iid if no URL is given" in result


@pytest.mark.asyncio
async def test_validate_work_item_url_with_invalid_url_without_work_item_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_work_item_url(
        url="https://example.com/namespace/project/-/work_items/42",
        group_id=None,
        project_id=None,
        work_item_iid=None,
    )
    assert isinstance(result, str)
    assert "Failed to parse work item URL" in result


@pytest.mark.asyncio
async def test_list_work_items_with_group_id(
    gitlab_client_mock, metadata, work_items_list
):
    graphql_response = {"namespace": {"workItems": {"nodes": work_items_list}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)

    response = await tool._arun(
        group_id="namespace/group",
        state="opened",
        search="test",
        author_username="test_user",
    )

    expected_response = json.dumps({"work_items": work_items_list})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_list_work_items_with_project_id(
    gitlab_client_mock, metadata, work_items_list
):
    graphql_response = {"project": {"workItems": {"nodes": work_items_list}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        state="opened",
        search="test",
        author_username="test_user",
    )

    expected_response = json.dumps({"work_items": work_items_list})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_list_work_items_with_group_url(
    gitlab_client_mock, metadata, work_items_list
):
    resolved_parent = ResolvedParent(type="group", full_path="namespace/group")
    graphql_response = {"namespace": {"workItems": {"nodes": work_items_list}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    response = await tool._arun(
        url="https://gitlab.com/groups/namespace/group", state="opened"
    )

    expected_response = json.dumps({"work_items": work_items_list})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_list_work_items_with_project_url(
    gitlab_client_mock, metadata, work_items_list
):
    graphql_response = {"project": {"workItems": {"nodes": work_items_list}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project", state="opened"
    )

    expected_response = json.dumps({"work_items": work_items_list})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_list_work_items_with_invalid_url(gitlab_client_mock, metadata):
    tool = ListWorkItems(description="list work items", metadata=metadata)

    response = await tool._arun(url="https://example.com/not-gitlab")

    response_json = json.loads(response)
    assert "error" in response_json
    assert (
        "Failed to parse parent work item URL: URL netloc 'example.com' does not match gitlab_host 'gitlab.com'"
        in response_json["error"]
    )
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_list_work_items_with_graphql_error(gitlab_client_mock, metadata):
    gitlab_client_mock.graphql = AsyncMock(side_effect=Exception("GraphQL error"))

    tool = ListWorkItems(description="list work items", metadata=metadata)

    response = await tool._arun(group_id="namespace/group")

    expected_response = json.dumps({"error": "GraphQL error"})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ListWorkItemsInput(group_id="namespace/group"),
            "List work items in group namespace/group",
        ),
        (
            ListWorkItemsInput(project_id="namespace/project"),
            "List work items in project namespace/project",
        ),
        (
            ListWorkItemsInput(url="https://gitlab.com/groups/namespace/group"),
            "List work items in https://gitlab.com/groups/namespace/group",
        ),
    ],
)
def test_list_work_items_format_display_message(input_data, expected_message):
    tool = ListWorkItems(description="list work items")
    message = tool.format_display_message(input_data)
    assert message == expected_message


@pytest.mark.asyncio
async def test_get_work_item_with_group_id(
    gitlab_client_mock, metadata, work_item_data
):
    graphql_response = {"namespace": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(group_id="namespace/group", work_item_iid=42)

    expected_response = json.dumps({"work_item": work_item_data})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_with_project_id(
    gitlab_client_mock, metadata, work_item_data
):
    graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"work_item": work_item_data})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_with_group_url(
    gitlab_client_mock, metadata, work_item_data
):
    graphql_response = {"namespace": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    resolved_item = ResolvedWorkItem(
        parent=ResolvedParent(type="group", full_path="namespace/group"),
        work_item_iid=42,
    )
    tool._validate_work_item_url = AsyncMock(return_value=resolved_item)

    result = await tool._arun(
        url="https://gitlab.com/groups/namespace/group/-/work_items/42"
    )

    expected = json.dumps({"work_item": work_item_data})
    assert result == expected

    tool._validate_work_item_url.assert_called_once_with(
        url="https://gitlab.com/groups/namespace/group/-/work_items/42",
        group_id=None,
        project_id=None,
        work_item_iid=None,
    )

    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args[0]
    assert "namespace" in call_args[1]["fullPath"]
    assert call_args[1]["iid"] == "42"


@pytest.mark.asyncio
async def test_get_work_item_with_project_url(
    gitlab_client_mock, metadata, work_item_data
):
    graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(
        url="https://gitlab.com/namespace/project/-/work_items/42"
    )

    expected_response = json.dumps({"work_item": work_item_data})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_not_found(gitlab_client_mock, metadata):
    graphql_response = {"project": {"workItems": {"nodes": []}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=999)

    expected_response = json.dumps({"work_item": None})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_with_graphql_error(gitlab_client_mock, metadata):
    gitlab_client_mock.graphql = AsyncMock(side_effect=Exception("GraphQL error"))

    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"error": "GraphQL error"})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_with_invalid_url(gitlab_client_mock, metadata):
    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(url="https://gitlab.com/invalid-url")

    response_json = json.loads(response)
    assert "error" in response_json
    assert "URL is not a work item URL" in response_json["error"]
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_get_work_item_with_no_iid(gitlab_client_mock, metadata):
    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(project_id="namespace/project")

    response_json = json.loads(response)
    assert "error" in response_json
    assert "Must provide work_item_iid if no URL is given" in response_json["error"]
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_get_work_item_missing_root_key(gitlab_client_mock, metadata):
    graphql_response = {}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    response_json = json.loads(response)
    assert "error" in response_json
    assert "No project found in response" in response_json["error"]

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            WorkItemResourceInput(group_id="namespace/group", work_item_iid=42),
            "Read work item #42 in group namespace/group",
        ),
        (
            WorkItemResourceInput(project_id="namespace/project", work_item_iid=42),
            "Read work item #42 in project namespace/project",
        ),
        (
            WorkItemResourceInput(
                url="https://gitlab.com/namespace/project/-/work_items/42"
            ),
            "Read work item https://gitlab.com/namespace/project/-/work_items/42",
        ),
    ],
)
def test_get_work_item_format_display_message(input_data, expected_message):
    tool = GetWorkItem(description="get work item")
    message = tool.format_display_message(input_data)
    assert message == expected_message


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


@pytest.mark.asyncio
async def test_get_work_item_notes_with_group_id(
    gitlab_client_mock, metadata, work_item_notes
):
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

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_project_id(
    gitlab_client_mock, metadata, work_item_notes
):
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

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_group_url(
    gitlab_client_mock, metadata, work_item_notes
):
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

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_project_url(
    gitlab_client_mock, metadata, work_item_notes
):
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

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_no_widgets(gitlab_client_mock, metadata):
    graphql_response = {"project": {"workItems": {"nodes": [{"widgets": []}]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"notes": []})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_empty_notes(gitlab_client_mock, metadata):
    graphql_response = {
        "project": {"workItems": {"nodes": [{"widgets": [{"notes": {"nodes": []}}]}]}}
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"notes": []}, indent=2)
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_not_found(gitlab_client_mock, metadata):
    graphql_response = {"project": {"workItems": {"nodes": []}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=999)

    expected_response = json.dumps({"error": "No work item found."})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_graphql_error(gitlab_client_mock, metadata):
    gitlab_client_mock.graphql = AsyncMock(side_effect=Exception("GraphQL error"))

    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(project_id="namespace/project", work_item_iid=42)

    expected_response = json.dumps({"error": "GraphQL error"})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_get_work_item_notes_with_invalid_url(gitlab_client_mock, metadata):
    tool = GetWorkItemNotes(description="get work item notes", metadata=metadata)

    response = await tool._arun(url="https://gitlab.com/invalid-url")

    response_json = json.loads(response)
    assert "error" in response_json
    assert "URL is not a work item URL" in response_json["error"]
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


@pytest.fixture
def created_work_item_data():
    """Fixture for created work item data."""
    return {
        "id": "gid://gitlab/WorkItem/123",
        "iid": "42",
        "title": "New Work Item",
        "description": "This is a newly created work item",
        "state": "opened",
        "createdAt": "2025-04-29T11:35:36.000+02:00",
        "author": {"username": "test_user", "name": "Test User"},
    }


@pytest.fixture
def work_item_type_data():
    """Fixture for work item type data."""
    return {
        "namespace": {
            "workItemTypes": {
                "nodes": [
                    {
                        "id": "gid://gitlab/WorkItems::Type/1",
                        "name": "Issue",
                    },
                    {
                        "id": "gid://gitlab/WorkItems::Type/2",
                        "name": "Epic",
                    },
                    {
                        "id": "gid://gitlab/WorkItems::Type/3",
                        "name": "Task",
                    },
                ]
            }
        }
    }


@pytest.mark.asyncio
async def test_create_work_item_with_group_id(
    gitlab_client_mock, metadata, created_work_item_data, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data,
        {"workItemCreate": {"workItem": created_work_item_data, "errors": []}},
    ]

    tool = CreateWorkItem(description="create work item", metadata=metadata)

    response = await tool._arun(
        group_id="namespace/group",
        title="New Work Item",
        type_name="Issue",
        description="This is a description",
    )

    response_json = json.loads(response)
    assert "work_item" in response_json
    assert response_json["work_item"] == created_work_item_data
    assert "message" in response_json
    assert "created successfully" in response_json["message"]

    # Verify graphql was called with correct parameters
    assert gitlab_client_mock.graphql.call_count == 2
    # First call to get work item types
    first_call_args = gitlab_client_mock.graphql.call_args_list[0][0]
    assert "workItemTypes" in first_call_args[0]
    # Second call to create work item
    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "workItemCreate" in second_call_args[0]
    assert second_call_args[1]["input"]["title"] == "New Work Item"
    assert second_call_args[1]["input"]["namespacePath"] == "namespace/group"


@pytest.mark.asyncio
async def test_create_work_item_with_all_supported_widgets(
    gitlab_client_mock, metadata, created_work_item_data, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data,
        {"workItemCreate": {"workItem": created_work_item_data, "errors": []}},
    ]

    tool = CreateWorkItem(description="create work item", metadata=metadata)
    tool._validate_parent_url = AsyncMock(
        return_value=ResolvedParent(type="group", full_path="namespace/group")
    )

    response = await tool._arun(
        group_id="namespace/group",
        title="Full Widget Test",
        type_name="Issue",
        description="Testing all supported widgets",
        assignee_ids=[123, 456],
        label_ids=["789", "101"],
        confidential=True,
        start_date="2025-07-01",
        due_date="2025-07-10",
        is_fixed=True,
        health_status="onTrack",
    )

    response_json = json.loads(response)
    assert "work_item" in response_json
    assert "message" in response_json
    gql_input = gitlab_client_mock.graphql.call_args_list[1][0][1]["input"]

    assert gql_input["confidential"] is True
    assert gql_input["assigneesWidget"]["assigneeIds"] == [
        "gid://gitlab/User/123",
        "gid://gitlab/User/456",
    ]
    assert gql_input["labelsWidget"]["labelIds"] == [
        "gid://gitlab/Label/789",
        "gid://gitlab/Label/101",
    ]
    assert gql_input["startAndDueDateWidget"] == {
        "startDate": "2025-07-01",
        "dueDate": "2025-07-10",
        "isFixed": True,
    }
    assert gql_input["healthStatusWidget"]["healthStatus"] == "onTrack"


@pytest.mark.asyncio
async def test_create_work_item_with_group_url(
    gitlab_client_mock, metadata, created_work_item_data, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data,
        {"workItemCreate": {"workItem": created_work_item_data, "errors": []}},
    ]
    tool = CreateWorkItem(description="create work item", metadata=metadata)

    resolved_parent = ResolvedParent(type="group", full_path="namespace/group")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    response = await tool._arun(
        url="https://gitlab.com/groups/namespace/group",
        title="New Work Item",
        type_name="Epic",
        health_status="onTrack",
    )

    response_json = json.loads(response)
    assert "work_item" in response_json
    assert response_json["work_item"] == created_work_item_data
    assert "message" in response_json

    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "healthStatusWidget" in second_call_args[1]["input"]
    assert (
        second_call_args[1]["input"]["healthStatusWidget"]["healthStatus"] == "onTrack"
    )


@pytest.mark.asyncio
async def test_create_work_item_with_project_id(
    gitlab_client_mock, metadata, created_work_item_data, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data,
        {"workItemCreate": {"workItem": created_work_item_data, "errors": []}},
    ]

    tool = CreateWorkItem(description="create work item", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        title="New Task",
        type_name="Task",
        description="Project-level work item",
    )

    response_json = json.loads(response)
    assert "work_item" in response_json
    assert response_json["work_item"] == created_work_item_data
    assert "message" in response_json

    gql_input = gitlab_client_mock.graphql.call_args_list[1][0][1]["input"]
    assert gql_input["title"] == "New Task"
    assert gql_input["namespacePath"] == "namespace/project"


@pytest.mark.asyncio
async def test_create_work_item_with_project_url(
    gitlab_client_mock, metadata, created_work_item_data, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data,
        {"workItemCreate": {"workItem": created_work_item_data, "errors": []}},
    ]

    tool = CreateWorkItem(description="create work item", metadata=metadata)
    tool._validate_parent_url = AsyncMock(
        return_value=ResolvedParent(type="project", full_path="namespace/project")
    )

    response = await tool._arun(
        url="https://gitlab.com/namespace/project",
        title="Work Item via URL",
        type_name="Task",
    )

    response_json = json.loads(response)
    assert "work_item" in response_json
    assert response_json["work_item"] == created_work_item_data
    assert "message" in response_json

    gql_input = gitlab_client_mock.graphql.call_args_list[1][0][1]["input"]
    assert gql_input["namespacePath"] == "namespace/project"


@pytest.mark.asyncio
async def test_create_work_item_with_error_response(
    gitlab_client_mock, metadata, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data,
        {
            "workItemCreate": {
                "workItem": None,
                "errors": ["Title cannot be blank"],
            }
        },
    ]

    tool = CreateWorkItem(description="create work item", metadata=metadata)

    resolved_parent = ResolvedParent(type="group", full_path="namespace/group")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    response = await tool._arun(
        group_id="namespace/group",
        title="",  # Empty title to trigger error
        type_name="Issue",
    )

    response_json = json.loads(response)
    assert "error" in response_json
    assert "details" in response_json
    assert response_json["details"]["work_item_errors"] == ["Title cannot be blank"]


@pytest.mark.asyncio
async def test_create_work_item_invalid_type(
    gitlab_client_mock, metadata, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [work_item_type_data]

    tool = CreateWorkItem(description="create work item", metadata=metadata)

    resolved_parent = ResolvedParent(type="group", full_path="namespace/group")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    response = await tool._arun(
        group_id="namespace/group",
        title="New Work Item",
        type_name="invalid_type",  # Type that doesn't exist
    )

    response_json = json.loads(response)
    assert "error" in response_json
    assert "Unknown work item type: 'invalid_type'" in response_json["error"]


@pytest.mark.asyncio
async def test_create_epic_in_project_error(
    gitlab_client_mock, metadata, work_item_type_data
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [work_item_type_data]

    tool = CreateWorkItem(description="create work item", metadata=metadata)

    resolved_parent = ResolvedParent(type="project", full_path="namespace/project")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    response = await tool._arun(
        project_id="namespace/project",
        title="New Epic",
        type_name="Epic",  # Epics can only be created in groups
    )

    response_json = json.loads(response)
    assert "error" in response_json
    assert (
        "Work item type 'Epic' cannot be created in a project – only in groups."
        in response_json["error"]
    )


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            CreateWorkItemInput(
                group_id="namespace/group", title="Test Item", type_name="Issue"
            ),
            "Create work item 'Test Item' in group namespace/group",
        ),
        (
            CreateWorkItemInput(
                project_id="namespace/project", title="Test Item", type_name="Task"
            ),
            "Create work item 'Test Item' in project namespace/project",
        ),
    ],
)
def test_create_work_item_format_display_message(input_data, expected_message):
    tool = CreateWorkItem(description="create work item")
    message = tool.format_display_message(input_data)
    assert message == expected_message
