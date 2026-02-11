# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.tools.work_item import ListWorkItems, ListWorkItemsInput
from duo_workflow_service.tools.work_items.base_tool import ResolvedParent


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
        "assignee_usernames": ["admin"],
        "health_status_filter": "atRisk",
        "status": {"name": "Won't do"},
        "milestone_title": ["milestone1"],
        "milestone_wildcard_id": "ANY",
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
        "assigneeUsernames": ["admin"],
        "healthStatusFilter": "atRisk",
        "status": {"name": "Won't do"},
        "milestoneTitle": ["milestone1"],
        "milestoneWildcardId": "ANY",
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


@pytest.mark.asyncio
@patch("duo_workflow_service.tools.work_item.get_query_variables_for_version")
async def test_list_work_items_calls_version_compatibility(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
):
    mock_get_query_variables.return_value = {"includeHierarchyWidget": True}
    tool = ListWorkItems(description="list work items", metadata=metadata)

    await tool._arun(project_id="namespace/project")

    mock_get_query_variables.assert_called_once_with("includeHierarchyWidget")
    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["includeHierarchyWidget"] is True


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
