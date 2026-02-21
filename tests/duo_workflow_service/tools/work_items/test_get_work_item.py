# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.work_item import GetWorkItem, WorkItemResourceInput
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


@pytest.fixture(name="work_item_data")
def work_item_data_fixture():
    """Fixture for sample work item data."""
    return {
        "id": "gid://gitlab/WorkItem/123",
        "iid": "42",
        "title": "Test Work Item",
        "description": "This is a test work item",
        "state": "opened",
        "confidential": False,
        "createdAt": "2025-04-29T11:35:36.000+02:00",
        "updatedAt": "2025-04-29T12:35:36.000+02:00",
        "author": {"username": "test_user", "name": "Test User"},
        "workItemType": {"name": "Issue"},
    }


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
