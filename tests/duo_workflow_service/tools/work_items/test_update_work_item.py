# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.work_item import (
    TodoAction,
    UpdateWorkItem,
    UpdateWorkItemInput,
)
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
        "createdAt": "2025-04-29T11:35:36.000+02:00",
        "updatedAt": "2025-04-29T12:35:36.000+02:00",
        "author": {"username": "test_user", "name": "Test User"},
        "workItemType": {"name": "Issue"},
    }


@pytest.fixture(name="resolved_work_item_fixture")
def resolved_work_item_fixture_func(work_item_data):
    return ResolvedWorkItem(
        id="gid://gitlab/WorkItem/123",
        full_data=work_item_data,
        parent=ResolvedParent(type="project", full_path="namespace/project"),
    )


@pytest.fixture(name="update_response_fixture")
def update_response_fixture_func():
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
        (
            {"todo_action": "add"},
            {"currentUserTodosWidget": {"action": "ADD"}},
        ),
        (
            {
                "todo_action": "mark_as_done",
                "todo_id": "gid://gitlab/Todo/123",
            },
            {
                "currentUserTodosWidget": {
                    "action": "MARK_AS_DONE",
                    "todoId": "gid://gitlab/Todo/123",
                }
            },
        ),
        (
            {"todo_action": "mark_as_done"},
            {"currentUserTodosWidget": {"action": "MARK_AS_DONE"}},
        ),
        (
            {"weight": 5},
            {"weightWidget": {"weight": 5}},
        ),
        (
            {"weight": 0},
            {"weightWidget": {"weight": 0}},
        ),
        (
            {"clear_weight": True},
            {"weightWidget": {"weight": None}},
        ),
        (
            {
                "agent_plan": "## Why\n\nReason\n\n## What\n\nSolution\n\n## How\n\nApproach"
            },
            {
                "agentPlanWidget": {
                    "content": "## Why\n\nReason\n\n## What\n\nSolution\n\n## How\n\nApproach"
                }
            },
        ),
    ],
    ids=[
        "title",
        "title_and_confidential",
        "close_state",
        "start_and_due_dates",
        "health_status",
        "assignees",
        "labels",
        "todo_add",
        "todo_mark_as_done_with_id",
        "todo_mark_as_done_without_id",
        "weight_set",
        "weight_zero_literal",
        "weight_clear",
        "agent_plan",
    ],
)
async def test_update_work_item_variants(
    gitlab_client_mock,
    metadata,
    resolved_work_item_fixture,
    update_response_fixture,
    update_kwargs,
    expected_fields,
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        **update_kwargs,
    )

    expected_output = json.dumps(
        {
            "updated_work_item": update_response_fixture["data"]["workItemUpdate"][
                "workItem"
            ]
        }
    )
    assert result == expected_output

    mutation, variables = gitlab_client_mock.graphql.call_args[0]
    assert "workItemUpdate" in mutation

    input_data = variables["input"]
    for key, value in expected_fields.items():
        assert input_data[key] == value


@pytest.mark.asyncio
async def test_update_work_item_with_group_id(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        group_id="namespace/group",
        work_item_iid=42,
        title="Updated Title",
    )

    assert json.loads(result)["updated_work_item"]["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_update_work_item_with_project_id(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        title="Updated Title",
    )

    assert json.loads(result)["updated_work_item"]["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_update_work_item_with_url(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        url="https://gitlab.com/namespace/project/-/work_items/42",
        title="Updated Title",
    )

    assert json.loads(result)["updated_work_item"]["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_update_work_item_graphql_error(
    gitlab_client_mock, metadata, resolved_work_item_fixture
):
    graphql_response = {"errors": [{"message": "Invalid field"}]}
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            title="Trigger error",
        )

    assert "Invalid field" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_work_item_validation_error(gitlab_client_mock, metadata):
    # Return empty nodes list to trigger "Work item not found" error
    graphql_response = {"project": {"workItems": {"nodes": []}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = UpdateWorkItem(description="update", metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            title="Bad",
        )

    assert "Work item" in str(exc_info.value) and "not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_update_work_item_exception(
    gitlab_client_mock, metadata, resolved_work_item_fixture
):
    gitlab_client_mock.graphql = AsyncMock(side_effect=Exception("Network error"))
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)

    with pytest.raises(Exception, match="Network error"):
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            title="Trigger exception",
        )

    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_update_work_item_invalid_work_item(gitlab_client_mock, metadata):
    # Return empty nodes list to trigger "Work item not found" error
    graphql_response = {"project": {"workItems": {"nodes": []}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = UpdateWorkItem(description="update work item", metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._arun(
            project_id="namespace/project",
            work_item_iid=999,
            title="This update will fail",
        )

    assert "Work item" in str(exc_info.value) and "not found" in str(exc_info.value)
    gitlab_client_mock.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_update_work_item_with_hierarchy_widget_unique(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/789"},
    )

    expected_output = json.dumps(
        {
            "updated_work_item": update_response_fixture["data"]["workItemUpdate"][
                "workItem"
            ]
        }
    )
    assert result == expected_output

    call_args = gitlab_client_mock.graphql.call_args[0]
    assert "workItemUpdate" in call_args[0]

    input_data = call_args[1]["input"]
    assert "hierarchyWidget" in input_data
    assert input_data["hierarchyWidget"]["parentId"] == "gid://gitlab/WorkItem/789"


@pytest.mark.asyncio
async def test_update_work_item_with_invalid_hierarchy_widget_unique(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

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
    call_args = gitlab_client_mock.graphql.call_args[0]
    input_data = call_args[1]["input"]
    assert "hierarchyWidget" not in input_data


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


def test_update_work_item_input_with_valid_hierarchy_widget():
    """Test UpdateWorkItemInput validation with valid hierarchy_widget."""
    input_data = UpdateWorkItemInput(
        group_id="test/group",
        work_item_iid=42,
        hierarchy_widget={"parent_id": "gid://gitlab/WorkItem/456"},
    )

    assert input_data.hierarchy_widget == {"parent_id": "gid://gitlab/WorkItem/456"}
    assert input_data.work_item_iid == 42


def test_update_work_item_input_without_hierarchy_widget():
    """Test UpdateWorkItemInput validation without hierarchy_widget."""
    input_data = UpdateWorkItemInput(group_id="test/group", work_item_iid=42)

    assert input_data.hierarchy_widget is None
    assert input_data.work_item_iid == 42


def test_update_work_item_input_with_todo_action_add():
    """Test UpdateWorkItemInput accepts todo_action='add'."""
    input_data = UpdateWorkItemInput(
        project_id="namespace/project",
        work_item_iid=42,
        todo_action="add",
    )
    assert input_data.todo_action == TodoAction.ADD
    assert input_data.todo_id is None


def test_update_work_item_input_with_todo_action_mark_as_done():
    """Test UpdateWorkItemInput accepts todo_action='mark_as_done' with todo_id."""
    input_data = UpdateWorkItemInput(
        project_id="namespace/project",
        work_item_iid=42,
        todo_action="mark_as_done",
        todo_id="gid://gitlab/Todo/123",
    )
    assert input_data.todo_action == TodoAction.MARK_AS_DONE
    assert input_data.todo_id == "gid://gitlab/Todo/123"


def test_update_work_item_input_without_todo_fields():
    """Test UpdateWorkItemInput defaults todo fields to None."""
    input_data = UpdateWorkItemInput(
        project_id="namespace/project",
        work_item_iid=42,
    )
    assert input_data.todo_action is None
    assert input_data.todo_id is None


@pytest.mark.asyncio
async def test_update_work_item_with_invalid_todo_id(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    """Test that invalid todo_id format produces a warning but still succeeds."""
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        todo_action="mark_as_done",
        todo_id="bad-format",
    )

    response_json = json.loads(result)
    assert "updated_work_item" in response_json
    assert "warnings" in response_json
    assert any("Invalid todo_id format" in w for w in response_json["warnings"])

    # Verify the widget was sent without todoId
    call_args = gitlab_client_mock.graphql.call_args[0]
    input_data = call_args[1]["input"]
    assert "currentUserTodosWidget" in input_data
    assert "todoId" not in input_data["currentUserTodosWidget"]


@pytest.mark.asyncio
async def test_update_work_item_with_invalid_todo_action(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    """Test that invalid todo_action produces a warning and skips the widget."""
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        todo_action="invalid_action",
    )

    response_json = json.loads(result)
    assert "updated_work_item" in response_json
    assert "warnings" in response_json
    assert any("Invalid todo_action" in w for w in response_json["warnings"])

    # Verify the widget was NOT included
    call_args = gitlab_client_mock.graphql.call_args[0]
    input_data = call_args[1]["input"]
    assert "currentUserTodosWidget" not in input_data


@pytest.mark.asyncio
async def test_update_work_item_without_todo_action(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    """Test that omitting todo_action does not include the currentUserTodosWidget."""
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        title="Updated Title",
    )

    response_json = json.loads(result)
    assert "updated_work_item" in response_json

    # Verify the widget was NOT included when todo_action is not provided
    call_args = gitlab_client_mock.graphql.call_args[0]
    input_data = call_args[1]["input"]
    assert "currentUserTodosWidget" not in input_data


@pytest.mark.asyncio
async def test_update_work_item_add_action_drops_todo_id(
    gitlab_client_mock, metadata, resolved_work_item_fixture, update_response_fixture
):
    """Test that todo_id is dropped with a warning when todo_action is 'add'."""
    tool = UpdateWorkItem(description="update", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item_fixture)
    gitlab_client_mock.graphql = AsyncMock(return_value=update_response_fixture)

    result = await tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        todo_action="add",
        todo_id="gid://gitlab/Todo/456",
    )

    response_json = json.loads(result)
    assert "updated_work_item" in response_json
    assert "warnings" in response_json
    assert any(
        "todo_id is only applicable when todo_action is 'mark_as_done'." in w
        for w in response_json["warnings"]
    )

    # Verify the widget was sent as ADD without todoId
    call_args = gitlab_client_mock.graphql.call_args[0]
    input_data = call_args[1]["input"]
    assert input_data["currentUserTodosWidget"] == {"action": "ADD"}
    assert "todoId" not in input_data["currentUserTodosWidget"]
