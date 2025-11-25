import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.work_item import (
    CreateWorkItem,
    CreateWorkItemInput,
    CreateWorkItemNote,
    CreateWorkItemNoteInput,
    GetWorkItem,
    GetWorkItemNotes,
    GetWorkItemNotesInput,
    ListWorkItems,
    ListWorkItemsInput,
    UpdateWorkItem,
    UpdateWorkItemInput,
    WorkItemResourceInput,
)
from duo_workflow_service.tools.work_items.base_tool import (
    ResolvedParent,
    ResolvedWorkItem,
    WorkItemBaseTool,
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

    expected_response = json.dumps({"work_items": work_items_list, "page_info": {}})
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

    expected_response = json.dumps({"work_items": work_items_list, "page_info": {}})
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

    expected_response = json.dumps({"work_items": work_items_list, "page_info": {}})
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

    expected_response = json.dumps({"work_items": work_items_list, "page_info": {}})
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
@pytest.mark.parametrize(
    "parent_type, parent_key, parent_id",
    [
        ("group", "namespace", "namespace/group"),
        ("project", "project", "namespace/project"),
    ],
)
async def test_list_work_items_with_filters(
    gitlab_client_mock, metadata, work_items_list, parent_type, parent_key, parent_id
):
    graphql_response = {
        parent_key: {
            "workItems": {
                "nodes": work_items_list,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)

    args = {
        f"{parent_type}_id": parent_id,
        "state": "opened",
        "author_username": "johndoe",
        "search": "bug",
        "created_after": "2025-01-01T00:00:00Z",
        "created_before": "2025-06-01T00:00:00Z",
        "updated_after": "2025-02-01T00:00:00Z",
        "updated_before": "2025-11-01T00:00:00Z",
        "due_after": "2025-03-01T00:00:00Z",
        "due_before": "2025-12-31T00:00:00Z",
        "sort": "CREATED_DESC",
        "label_name": ["api"],
    }

    response = await tool._arun(**args)

    response_json = json.loads(response)
    assert response_json["work_items"] == work_items_list
    assert response_json["page_info"] == {"hasNextPage": False, "endCursor": None}

    expected_vars = {
        "fullPath": parent_id,
        "state": "opened",
        "authorUsername": "johndoe",
        "search": "bug",
        "createdAfter": "2025-01-01T00:00:00Z",
        "createdBefore": "2025-06-01T00:00:00Z",
        "updatedAfter": "2025-02-01T00:00:00Z",
        "updatedBefore": "2025-11-01T00:00:00Z",
        "dueAfter": "2025-03-01T00:00:00Z",
        "dueBefore": "2025-12-31T00:00:00Z",
        "sort": "CREATED_DESC",
        "labelName": ["api"],
    }

    gql_vars = gitlab_client_mock.graphql.call_args[0][1]
    # Only check variables that are present in both dictionaries
    for key, expected_value in expected_vars.items():
        assert (
            key in gql_vars
        ), f"Expected variable {key} not found in GraphQL variables"
        assert (
            gql_vars[key] == expected_value
        ), f"Expected {key}={expected_value}, got {gql_vars[key]}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "types_input,expected_types,warnings_expected",
    [
        (["Issue", "Epic"], ["ISSUE", "EPIC"], False),
        (["Banana", "Task"], ["TASK"], True),
        (["invalid1", "invalid2"], [], True),
    ],
)
async def test_list_work_items_with_types_filtering(
    gitlab_client_mock,
    metadata,
    work_items_list,
    types_input,
    expected_types,
    warnings_expected,
):
    graphql_response = {
        "project": {
            "workItems": {
                "nodes": work_items_list,
                "pageInfo": {"hasNextPage": False, "endCursor": None},
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)

    args = {
        "project_id": "namespace/project",
        "types": types_input,
    }

    response = await tool._arun(**args)
    response_json = json.loads(response)

    assert response_json["work_items"] == work_items_list

    gql_vars = gitlab_client_mock.graphql.call_args[0][1]
    if expected_types:
        assert gql_vars["types"] == expected_types
    else:
        assert "types" not in gql_vars

    if warnings_expected:
        assert "warnings" in response_json
        assert "Some types were invalid" in response_json["warnings"][0]
    else:
        assert "warnings" not in response_json


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_type, parent_key, parent_id",
    [
        ("group", "namespace", "namespace/group"),
        ("project", "project", "namespace/project"),
    ],
)
async def test_list_work_items_with_pagination(
    gitlab_client_mock, metadata, work_items_list, parent_type, parent_key, parent_id
):
    graphql_response = {
        parent_key: {
            "workItems": {
                "nodes": work_items_list,
                "pageInfo": {"hasNextPage": True, "endCursor": "abc123=="},
            }
        }
    }
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = ListWorkItems(description="list work items", metadata=metadata)

    args = {f"{parent_type}_id": parent_id, "first": 2, "after": "prev-cursor"}

    response = await tool._arun(**args)

    response_json = json.loads(response)
    assert response_json["work_items"] == work_items_list
    assert response_json["page_info"] == {"hasNextPage": True, "endCursor": "abc123=="}

    gql_vars = gitlab_client_mock.graphql.call_args[0][1]
    assert gql_vars["first"] == 2
    assert gql_vars["after"] == "prev-cursor"


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

    expected_response = json.dumps({"error": "Work item not found"})
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
    assert (
        "Failed to parse work item URL: Not a work item URL: https://gitlab.com/invalid-url"
        in response_json["error"]
    )
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


@pytest.mark.asyncio
async def test_create_work_item_with_hierarchy_widget(
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
        title="Child Work Item",
        type_name="Issue",
        hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/456"},
    )

    response_json = json.loads(response)
    assert "work_item" in response_json
    assert response_json["work_item"] == created_work_item_data
    assert "message" in response_json


@pytest.mark.asyncio
async def test_create_work_item_rejects_quick_actions_in_description(
    gitlab_client_mock, metadata
):
    tool = CreateWorkItem(description="create work item", metadata=metadata)

    response = await tool._arun(
        group_id="namespace/group",
        title="Blocked",
        type_name="Issue",
        description="/close",
    )

    resp = json.loads(response)
    assert "error" in resp
    gitlab_client_mock.graphql.assert_not_called()


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


@pytest.fixture
def created_note_data():
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
    created_note_data,
    params,
    response_key,
    expected_body,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        {response_key: {"workItems": {"nodes": [work_item_data]}}},
        {"createNote": {"note": created_note_data, "errors": []}},
    ]

    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)
    response = await tool._arun(**params)

    response_json = json.loads(response)
    assert response_json["status"] == "success"
    assert "message" in response_json
    assert "Note created successfully" in response_json["message"]
    assert response_json["note"] == created_note_data

    # Verify GraphQL was called twice
    assert gitlab_client_mock.graphql.call_count == 2

    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "createNote" in second_call_args[0]
    assert second_call_args[1]["input"]["noteableId"] == work_item_data["id"]
    assert second_call_args[1]["input"]["body"] == expected_body


@pytest.mark.asyncio
async def test_create_work_item_note_with_optional_parameters(
    gitlab_client_mock, metadata, work_item_data, created_note_data
):
    # Mock the GraphQL calls
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        {"project": {"workItems": {"nodes": [work_item_data]}}},
        {"createNote": {"note": created_note_data, "errors": []}},
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


@pytest.mark.asyncio
async def test_create_work_item_note_rejects_quick_actions_in_body(
    gitlab_client_mock, metadata
):
    tool = CreateWorkItemNote(description="create work item note", metadata=metadata)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        body="/close",
    )

    response_json = json.loads(response)
    assert "error" in response_json
    assert "Body contains GitLab quick actions" in response_json["error"]
    gitlab_client_mock.graphql.assert_not_called()


@pytest.fixture
def resolved_work_item(work_item_data):
    return ResolvedWorkItem(
        id="gid://gitlab/WorkItem/123",
        full_data=work_item_data,
        parent=ResolvedParent(type="project", full_path="namespace/project"),
    )


@pytest.fixture
def update_response():
    return {
        "data": {
            "workItemUpdate": {
                "workItem": {
                    "id": "gid://gitlab/WorkItem/123",
                    "title": "Updated Title",
                    "state": "opened",
                }
            }
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update_kwargs, expected_fields",
    [
        (
            {"title": "Updated Title"},
            {"title": "Updated Title"},
        ),
        (
            {"title": "Confidential Item", "confidential": True},
            {"title": "Confidential Item", "confidential": True},
        ),
        (
            {"state": "closed"},
            {"stateEvent": "CLOSE"},
        ),
        (
            {
                "start_date": "2025-08-01",
                "due_date": "2025-08-10",
                "is_fixed": True,
            },
            {
                "startAndDueDateWidget": {
                    "startDate": "2025-08-01",
                    "dueDate": "2025-08-10",
                    "isFixed": True,
                }
            },
        ),
        (
            {"health_status": "needsAttention"},
            {"healthStatusWidget": {"healthStatus": "needsAttention"}},
        ),
        (
            {"assignee_ids": [1, "gid://gitlab/User/2"]},
            {
                "assigneesWidget": {
                    "assigneeIds": ["gid://gitlab/User/1", "gid://gitlab/User/2"]
                }
            },
        ),
        (
            {"add_label_ids": [3], "remove_label_ids": ["gid://gitlab/Label/5"]},
            {
                "labelsWidget": {
                    "addLabelIds": ["gid://gitlab/Label/3"],
                    "removeLabelIds": ["gid://gitlab/Label/5"],
                }
            },
        ),
    ],
)
async def test_update_work_item_variants(
    gitlab_client_mock,
    metadata,
    resolved_work_item,
    update_response,
    update_kwargs,
    expected_fields,
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        **update_kwargs,
    )

    expected_output = json.dumps(
        {"updated_work_item": update_response["data"]["workItemUpdate"]["workItem"]}
    )
    assert result == expected_output

    mutation, variables = gitlab_client_mock.graphql.call_args[0]
    assert "workItemUpdate" in mutation

    input_data = variables["input"]
    for key, value in expected_fields.items():
        assert input_data[key] == value


@pytest.mark.asyncio
async def test_update_work_item_with_group_id(
    gitlab_client_mock, metadata, resolved_work_item, update_response
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response)

    result = await tool._arun(
        group_id="namespace/group",
        work_item_iid=42,
        title="Updated Title",
    )

    assert json.loads(result)["updated_work_item"]["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_update_work_item_with_project_id(
    gitlab_client_mock, metadata, resolved_work_item, update_response
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        title="Updated Title",
    )

    assert json.loads(result)["updated_work_item"]["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_update_work_item_with_url(
    gitlab_client_mock, metadata, resolved_work_item, update_response
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response)

    result = await tool._arun(
        url="https://gitlab.com/namespace/project/-/work_items/42",
        title="Updated Title",
    )

    assert json.loads(result)["updated_work_item"]["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_update_work_item_graphql_error(
    gitlab_client_mock, metadata, resolved_work_item
):
    graphql_response = {"errors": [{"message": "Invalid field"}]}
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        title="Trigger error",
    )

    assert json.loads(result)["error"] == graphql_response["errors"]


@pytest.mark.asyncio
async def test_update_work_item_validation_error(gitlab_client_mock, metadata):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value="Invalid reference")

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        title="Bad",
    )

    assert json.loads(result)["error"] == "Invalid reference"
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_update_work_item_exception(
    gitlab_client_mock, metadata, resolved_work_item
):
    gitlab_client_mock.graphql = AsyncMock(side_effect=Exception("Network error"))
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        title="Trigger exception",
    )

    expected = json.dumps({"error": "Network error"})
    assert response == expected
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_update_work_item_invalid_work_item(gitlab_client_mock, metadata):
    tool = UpdateWorkItem(description="update work item", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value="Work item not found")

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=999,
        title="This update will fail",
    )

    expected_response = json.dumps({"error": "Work item not found"})
    assert response == expected_response

    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_update_work_item_with_hierarchy_widget(
    gitlab_client_mock, metadata, resolved_work_item, update_response
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/789"},
    )

    expected_output = json.dumps(
        {"updated_work_item": update_response["data"]["workItemUpdate"]["workItem"]}
    )
    assert result == expected_output

    mutation, variables = gitlab_client_mock.graphql.call_args[0]
    assert "workItemUpdate" in mutation

    input_data = variables["input"]
    assert "hierarchyWidget" in input_data
    assert input_data["hierarchyWidget"]["parentId"] == "gid://gitlab/WorkItem/789"


@pytest.mark.asyncio
async def test_update_work_item_with_invalid_hierarchy_widget(
    gitlab_client_mock, metadata, resolved_work_item, update_response
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        hierarchy_widget={"parent_id": "invalid_format"},  # Invalid GID format
    )

    response_json = json.loads(result)
    assert "updated_work_item" in response_json
    assert "warnings" in response_json
    assert (
        "Invalid parent_id format: invalid_format. Expected GitLab GID."
        in response_json["warnings"]
    )

    # Verify hierarchy widget was not included in GraphQL input
    mutation, variables = gitlab_client_mock.graphql.call_args[0]
    input_data = variables["input"]
    assert "hierarchyWidget" not in input_data


@pytest.mark.asyncio
async def test_update_work_item_rejects_quick_actions_in_description(
    gitlab_client_mock, metadata, resolved_work_item
):
    tool = UpdateWorkItem(description="update work item", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)

    response = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        description="/close",
    )

    resp = json.loads(response)
    assert "error" in resp
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            UpdateWorkItemInput(group_id="namespace/group", work_item_iid=42),
            "Update work item #42 in group namespace/group",
        ),
        (
            UpdateWorkItemInput(project_id="namespace/project", work_item_iid=42),
            "Update work item #42 in project namespace/project",
        ),
        (
            UpdateWorkItemInput(
                url="https://gitlab.com/namespace/project/-/work_items/42"
            ),
            "Update work item in https://gitlab.com/namespace/project/-/work_items/42",
        ),
    ],
)
def test_update_work_item_format_display_message(input_data, expected_message):
    tool = UpdateWorkItem(description="update work item")
    message = tool.format_display_message(input_data)
    assert message == expected_message


class TestBuildWorkItemInputFields:
    """Test the _build_work_item_input_fields static method integration with hierarchy widget."""

    def test_build_work_item_input_fields_with_hierarchy_widget(self):
        """Test that _build_work_item_input_fields includes hierarchy widget."""
        kwargs = {
            "title": "Test Work Item",
            "type_name": "Issue",
            "hierarchy_widget": {"parent_id": "gid://gitlab/WorkItem/123"},
        }

        input_data, warnings = WorkItemBaseTool._build_work_item_input_fields(kwargs)

        assert input_data["title"] == "Test Work Item"
        assert "hierarchyWidget" in input_data
        assert input_data["hierarchyWidget"]["parentId"] == "gid://gitlab/WorkItem/123"
        assert warnings == []

    def test_build_work_item_input_fields_with_invalid_hierarchy_widget(self):
        """Test that _build_work_item_input_fields handles invalid hierarchy widget."""
        kwargs = {
            "title": "Test Work Item",
            "type_name": "Issue",
            "hierarchy_widget": {"parent_id": "invalid_format"},
        }

        input_data, warnings = WorkItemBaseTool._build_work_item_input_fields(kwargs)

        assert input_data["title"] == "Test Work Item"
        assert "hierarchyWidget" not in input_data
        assert (
            "Invalid parent_id format: invalid_format. Expected GitLab GID." in warnings
        )

    def test_build_work_item_input_fields_without_hierarchy_widget(self):
        """Test that _build_work_item_input_fields works without hierarchy widget."""
        kwargs = {
            "title": "Test Work Item",
            "type_name": "Issue",
        }

        input_data, warnings = WorkItemBaseTool._build_work_item_input_fields(kwargs)

        assert input_data["title"] == "Test Work Item"
        assert "hierarchyWidget" not in input_data
        assert warnings == []

    def test_build_work_item_input_fields_with_multiple_widgets(self):
        """Test that hierarchy widget works alongside other widgets."""
        kwargs = {
            "title": "Test Work Item",
            "type_name": "Issue",
            "assignee_ids": [123],
            "label_ids": ["456"],
            "hierarchy_widget": {"parent_id": "gid://gitlab/WorkItem/789"},
        }

        input_data, warnings = WorkItemBaseTool._build_work_item_input_fields(kwargs)

        assert input_data["title"] == "Test Work Item"
        assert "assigneesWidget" in input_data
        assert "labelsWidget" in input_data
        assert "hierarchyWidget" in input_data
        assert input_data["hierarchyWidget"]["parentId"] == "gid://gitlab/WorkItem/789"
        assert warnings == []


class TestWorkItemInputValidation:
    """Test Pydantic input validation for hierarchy_widget."""

    def test_create_work_item_input_with_valid_hierarchy_widget(self):
        """Test CreateWorkItemInput validation with valid hierarchy_widget."""
        input_data = CreateWorkItemInput(
            title="Test Item",
            type_name="Issue",
            group_id="test/group",
            hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/123"},
        )

        assert input_data.hierarchy_widget == {"parent_id": "gid://gitlab/WorkItem/123"}
        assert input_data.title == "Test Item"
        assert input_data.type_name == "Issue"

    def test_create_work_item_input_without_hierarchy_widget(self):
        """Test CreateWorkItemInput validation without hierarchy_widget."""
        input_data = CreateWorkItemInput(
            title="Test Item", type_name="Issue", group_id="test/group"
        )

        assert input_data.hierarchy_widget is None
        assert input_data.title == "Test Item"

    def test_update_work_item_input_with_valid_hierarchy_widget(self):
        """Test UpdateWorkItemInput validation with valid hierarchy_widget."""
        input_data = UpdateWorkItemInput(
            group_id="test/group",
            work_item_iid=42,
            hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/456"},
        )

        assert input_data.hierarchy_widget == {"parent_id": "gid://gitlab/WorkItem/456"}
        assert input_data.work_item_iid == 42

    def test_update_work_item_input_without_hierarchy_widget(self):
        """Test UpdateWorkItemInput validation without hierarchy_widget."""
        input_data = UpdateWorkItemInput(group_id="test/group", work_item_iid=42)

        assert input_data.hierarchy_widget is None
        assert input_data.work_item_iid == 42

    def test_hierarchy_widget_with_wrong_key_type_validation(self):
        """Test that Pydantic validates the hierarchy_widget structure."""
        # This should work - correct key
        input_data = CreateWorkItemInput(
            title="Test Item",
            type_name="Issue",
            group_id="test/group",
            hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/123"},
        )
        assert input_data.hierarchy_widget == {"parent_id": "gid://gitlab/WorkItem/123"}

    def test_hierarchy_widget_type_validation(self):
        """Test that hierarchy_widget must be a dict with specific structure."""
        # Test with valid structure
        input_data = CreateWorkItemInput(
            title="Test Item",
            type_name="Issue",
            group_id="test/group",
            hierarchy_widget={"parent_id": "some_value"},
        )
        assert input_data.hierarchy_widget == {"parent_id": "some_value"}
