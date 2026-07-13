# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.work_item import (
    GetWorkItemStatuses,
    ParentResourceInput,
)
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


@pytest.fixture(name="statuses_response")
def statuses_response_fixture():
    return {
        "namespace": {
            "lifecycles": {
                "nodes": [
                    {
                        "id": "gid://gitlab/WorkItems::Statuses::Lifecycle/1",
                        "name": "Default",
                        "statuses": [
                            {
                                "id": "gid://gitlab/WorkItems::Statuses::SystemDefined::Status/1",
                                "name": "To do",
                            },
                            {
                                "id": "gid://gitlab/WorkItems::Statuses::SystemDefined::Status/2",
                                "name": "In progress",
                            },
                            {
                                "id": "gid://gitlab/WorkItems::Statuses::SystemDefined::Status/3",
                                "name": "Done",
                            },
                        ],
                    }
                ]
            }
        }
    }


@pytest.mark.asyncio
async def test_get_work_item_statuses_with_group_id(
    gitlab_client_mock, metadata, statuses_response
):
    gitlab_client_mock.graphql = AsyncMock(return_value=statuses_response)
    tool = GetWorkItemStatuses(description="get statuses", metadata=metadata)
    tool._resolve_parent_path = AsyncMock(
        return_value=ResolvedParent(type="group", full_path="namespace/group")
    )

    result = await tool._arun(group_id="namespace/group")

    parsed = json.loads(result)
    assert parsed["lifecycles"] == statuses_response["namespace"]["lifecycles"]["nodes"]

    query, variables = gitlab_client_mock.graphql.call_args[0]
    assert "lifecycles" in query
    assert variables == {"fullPath": "namespace/group"}


@pytest.mark.asyncio
async def test_get_work_item_statuses_with_url(
    gitlab_client_mock, metadata, statuses_response
):
    gitlab_client_mock.graphql = AsyncMock(return_value=statuses_response)
    tool = GetWorkItemStatuses(description="get statuses", metadata=metadata)

    result = await tool._arun(url="https://gitlab.com/namespace/project")

    parsed = json.loads(result)
    assert len(parsed["lifecycles"]) == 1
    assert parsed["lifecycles"][0]["statuses"][1]["name"] == "In progress"


@pytest.mark.asyncio
async def test_get_work_item_statuses_with_project_id(
    gitlab_client_mock, metadata, statuses_response
):
    gitlab_client_mock.graphql = AsyncMock(return_value=statuses_response)
    tool = GetWorkItemStatuses(description="get statuses", metadata=metadata)
    tool._resolve_parent_path = AsyncMock(
        return_value=ResolvedParent(type="project", full_path="namespace/project")
    )

    result = await tool._arun(project_id="namespace/project")

    parsed = json.loads(result)
    assert parsed["lifecycles"] == statuses_response["namespace"]["lifecycles"]["nodes"]

    query, variables = gitlab_client_mock.graphql.call_args[0]
    assert "lifecycles" in query
    assert variables == {"fullPath": "namespace/project"}


@pytest.mark.asyncio
async def test_get_work_item_statuses_no_lifecycles(gitlab_client_mock, metadata):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"namespace": {"lifecycles": {"nodes": []}}}
    )
    tool = GetWorkItemStatuses(description="get statuses", metadata=metadata)
    tool._resolve_parent_path = AsyncMock(
        return_value=ResolvedParent(type="group", full_path="namespace/group")
    )

    result = await tool._arun(group_id="namespace/group")

    parsed = json.loads(result)
    assert parsed == {"lifecycles": []}


@pytest.mark.asyncio
async def test_get_work_item_statuses_graphql_error(gitlab_client_mock, metadata):
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"errors": [{"message": "Boom"}]}
    )
    tool = GetWorkItemStatuses(description="get statuses", metadata=metadata)
    tool._resolve_parent_path = AsyncMock(
        return_value=ResolvedParent(type="project", full_path="namespace/project")
    )

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_id="namespace/project")

    assert "Boom" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_work_item_statuses_no_namespace(gitlab_client_mock, metadata):
    gitlab_client_mock.graphql = AsyncMock(return_value={"namespace": None})
    tool = GetWorkItemStatuses(description="get statuses", metadata=metadata)
    tool._resolve_parent_path = AsyncMock(
        return_value=ResolvedParent(type="project", full_path="namespace/project")
    )

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(project_id="namespace/project")

    assert "No namespace found" in str(exc_info.value)


@pytest.mark.parametrize(
    "input_data,expected_message",
    [
        (
            ParentResourceInput(group_id="namespace/group"),
            "Get available work item statuses in group namespace/group",
        ),
        (
            ParentResourceInput(project_id="namespace/project"),
            "Get available work item statuses in project namespace/project",
        ),
        (
            ParentResourceInput(url="https://gitlab.com/namespace/group"),
            "Get available work item statuses in https://gitlab.com/namespace/group",
        ),
    ],
)
def test_get_work_item_statuses_format_display_message(input_data, expected_message):
    tool = GetWorkItemStatuses(description="get statuses")
    assert tool.format_display_message(input_data) == expected_message
