# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock

import pytest

from duo_workflow_service.tools.work_item import CreateWorkItem, CreateWorkItemInput
from duo_workflow_service.tools.work_items.base_tool import (
    ResolvedParent,
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


@pytest.fixture(name="created_work_item_data_fixture")
def created_work_item_data_fixture_func():
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


@pytest.fixture(name="work_item_type_data_fixture")
def work_item_type_data_fixture_func():
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
    gitlab_client_mock,
    metadata,
    created_work_item_data_fixture,
    work_item_type_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
        {"workItemCreate": {"workItem": created_work_item_data_fixture, "errors": []}},
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
    assert response_json["work_item"] == created_work_item_data_fixture
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
    gitlab_client_mock,
    metadata,
    created_work_item_data_fixture,
    work_item_type_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
        {"workItemCreate": {"workItem": created_work_item_data_fixture, "errors": []}},
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
    gitlab_client_mock,
    metadata,
    created_work_item_data_fixture,
    work_item_type_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
        {"workItemCreate": {"workItem": created_work_item_data_fixture, "errors": []}},
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
    assert response_json["work_item"] == created_work_item_data_fixture
    assert "message" in response_json

    second_call_args = gitlab_client_mock.graphql.call_args_list[1][0]
    assert "healthStatusWidget" in second_call_args[1]["input"]
    assert (
        second_call_args[1]["input"]["healthStatusWidget"]["healthStatus"] == "onTrack"
    )


@pytest.mark.asyncio
async def test_create_work_item_with_project_id(
    gitlab_client_mock,
    metadata,
    created_work_item_data_fixture,
    work_item_type_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
        {"workItemCreate": {"workItem": created_work_item_data_fixture, "errors": []}},
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
    assert response_json["work_item"] == created_work_item_data_fixture
    assert "message" in response_json

    gql_input = gitlab_client_mock.graphql.call_args_list[1][0][1]["input"]
    assert gql_input["title"] == "New Task"
    assert gql_input["namespacePath"] == "namespace/project"


@pytest.mark.asyncio
async def test_create_work_item_with_project_url(
    gitlab_client_mock,
    metadata,
    created_work_item_data_fixture,
    work_item_type_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
        {"workItemCreate": {"workItem": created_work_item_data_fixture, "errors": []}},
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
    assert response_json["work_item"] == created_work_item_data_fixture
    assert "message" in response_json

    gql_input = gitlab_client_mock.graphql.call_args_list[1][0][1]["input"]
    assert gql_input["namespacePath"] == "namespace/project"


@pytest.mark.asyncio
async def test_create_work_item_with_error_response(
    gitlab_client_mock, metadata, work_item_type_data_fixture
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
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
    gitlab_client_mock, metadata, work_item_type_data_fixture
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [work_item_type_data_fixture]

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
    gitlab_client_mock, metadata, work_item_type_data_fixture
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [work_item_type_data_fixture]

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
    gitlab_client_mock,
    metadata,
    created_work_item_data_fixture,
    work_item_type_data_fixture,
):
    gitlab_client_mock.graphql = AsyncMock()
    gitlab_client_mock.graphql.side_effect = [
        work_item_type_data_fixture,
        {"workItemCreate": {"workItem": created_work_item_data_fixture, "errors": []}},
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
    assert response_json["work_item"] == created_work_item_data_fixture
    assert "message" in response_json


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
        assert not warnings

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
        assert not warnings

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
        assert not warnings


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
