"""Tests for base tool methods shared across work item tools."""

# pylint: disable=file-naming-for-tests

from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.tools.work_item import GetWorkItem, UpdateWorkItem
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
@patch(
    "duo_workflow_service.tools.work_items.base_tool.get_query_variables_for_version"
)
async def test_fetch_work_item_data_calls_version_compatibility(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
):
    mock_get_query_variables.return_value = {"includeHierarchyWidget": True}
    work_item_data = {
        "id": "gid://gitlab/WorkItem/123",
        "iid": "42",
        "title": "Test Work Item",
    }
    graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = UpdateWorkItem(description="update work item", metadata=metadata)

    resolved = ResolvedWorkItem(
        parent=ResolvedParent(type="project", full_path="namespace/project"),
        work_item_iid=42,
    )

    await tool._fetch_work_item_data(resolved)

    mock_get_query_variables.assert_called_once_with("includeHierarchyWidget")
    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["includeHierarchyWidget"] is True
