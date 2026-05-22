# pylint: disable=file-naming-for-tests
"""Integration tests for version compatibility in work item tools."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.tools.work_item import (
    CreateWorkItem,
    GetWorkItem,
    ListWorkItems,
    UpdateWorkItem,
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
        "state": "opened",
        "createdAt": "2025-04-29T11:35:36.000+02:00",
        "updatedAt": "2025-04-29T12:35:36.000+02:00",
        "author": {"username": "test_user", "name": "Test User"},
    }


class TestGetWorkItemVersionCompatibility:
    """Tests for GetWorkItem with version compatibility."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_get_work_item_includes_hierarchy_widget_on_new_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test that includeHierarchyWidget is True for GitLab 18.7.0+."""
        mock_gitlab_version.get.return_value = "18.7.0"
        graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(project_id="namespace/project", work_item_iid=42)

        # Verify the GraphQL call includes the version compatibility variable
        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert "includeHierarchyWidget" in variables
        assert variables["includeHierarchyWidget"] is True

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_get_work_item_excludes_hierarchy_widget_on_old_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test that includeHierarchyWidget is False for GitLab < 18.7.0."""
        mock_gitlab_version.get.return_value = "18.6.0"
        graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(project_id="namespace/project", work_item_iid=42)

        # Verify the GraphQL call includes the version compatibility variable
        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert "includeHierarchyWidget" in variables
        assert variables["includeHierarchyWidget"] is False

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_get_work_item_with_group_id_on_new_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test GetWorkItem with group_id on new version."""
        mock_gitlab_version.get.return_value = "18.8.0"
        graphql_response = {"namespace": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(group_id="namespace/group", work_item_iid=42)

        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is True

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_get_work_item_fallback_when_version_unavailable(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test that fallback version is used when GitLab version is unavailable."""
        mock_gitlab_version.get.return_value = None
        graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(project_id="namespace/project", work_item_iid=42)

        # Fallback version is 18.6.0, which doesn't support hierarchy widget
        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is False


class TestListWorkItemsVersionCompatibility:
    """Tests for ListWorkItems with version compatibility."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_list_work_items_includes_hierarchy_widget_on_new_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata
    ):
        """Test that includeHierarchyWidget is True for GitLab 18.7.0+."""
        mock_gitlab_version.get.return_value = "18.7.0"
        work_items_list = [
            {
                "id": "gid://gitlab/WorkItem/123",
                "iid": "42",
                "title": "Test Work Item",
                "state": "opened",
            }
        ]
        graphql_response = {"project": {"workItems": {"nodes": work_items_list}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = ListWorkItems(description="list work items", metadata=metadata)
        await tool._arun(project_id="namespace/project")

        # Verify the GraphQL call includes the version compatibility variable
        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert "includeHierarchyWidget" in variables
        assert variables["includeHierarchyWidget"] is True

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_list_work_items_excludes_hierarchy_widget_on_old_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata
    ):
        """Test that includeHierarchyWidget is False for GitLab < 18.7.0."""
        mock_gitlab_version.get.return_value = "18.6.0"
        work_items_list = [
            {
                "id": "gid://gitlab/WorkItem/123",
                "iid": "42",
                "title": "Test Work Item",
                "state": "opened",
            }
        ]
        graphql_response = {"project": {"workItems": {"nodes": work_items_list}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = ListWorkItems(description="list work items", metadata=metadata)
        await tool._arun(project_id="namespace/project")

        # Verify the GraphQL call includes the version compatibility variable
        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert "includeHierarchyWidget" in variables
        assert variables["includeHierarchyWidget"] is False

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_list_work_items_with_group_id_on_new_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata
    ):
        """Test ListWorkItems with group_id on new version."""
        mock_gitlab_version.get.return_value = "19.0.0"
        work_items_list = [
            {
                "id": "gid://gitlab/WorkItem/123",
                "iid": "42",
                "title": "Test Work Item",
                "state": "opened",
            }
        ]
        graphql_response = {"namespace": {"workItems": {"nodes": work_items_list}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = ListWorkItems(description="list work items", metadata=metadata)
        await tool._arun(group_id="namespace/group")

        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is True

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_list_work_items_with_filters_and_version_compatibility(
        self, mock_gitlab_version, gitlab_client_mock, metadata
    ):
        """Test that version compatibility works alongside other filters."""
        mock_gitlab_version.get.return_value = "18.7.1"
        work_items_list = [
            {
                "id": "gid://gitlab/WorkItem/123",
                "iid": "42",
                "title": "Test Work Item",
                "state": "opened",
            }
        ]
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
        await tool._arun(
            project_id="namespace/project",
            state="opened",
            author_username="test_user",
            label_name=["bug"],
        )

        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]

        # Verify version compatibility variable is present
        assert variables["includeHierarchyWidget"] is True

        # Verify other filters are also present
        assert variables["state"] == "opened"
        assert variables["authorUsername"] == "test_user"
        assert variables["labelName"] == ["bug"]

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_list_work_items_fallback_when_version_invalid(
        self, mock_gitlab_version, gitlab_client_mock, metadata
    ):
        """Test that fallback version is used when GitLab version is invalid."""
        mock_gitlab_version.get.return_value = "invalid-version"
        work_items_list = [
            {
                "id": "gid://gitlab/WorkItem/123",
                "iid": "42",
                "title": "Test Work Item",
                "state": "opened",
            }
        ]
        graphql_response = {"project": {"workItems": {"nodes": work_items_list}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = ListWorkItems(description="list work items", metadata=metadata)
        await tool._arun(project_id="namespace/project")

        # Fallback version is 18.6.0, which doesn't support hierarchy widget
        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is False


class TestVersionCompatibilityEdgeCases:
    """Tests for edge cases in version compatibility."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_exact_threshold_version_supports_hierarchy(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test that exact threshold version (18.7.0) supports hierarchy widget."""
        mock_gitlab_version.get.return_value = "18.7.0"
        graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(project_id="namespace/project", work_item_iid=42)

        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is True

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_just_below_threshold_does_not_support_hierarchy(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test that version just below threshold (18.6.9) doesn't support hierarchy widget."""
        mock_gitlab_version.get.return_value = "18.6.9"
        graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(project_id="namespace/project", work_item_iid=42)

        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is False

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_patch_version_above_threshold_supports_hierarchy(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_data
    ):
        """Test that patch version above threshold (18.7.1) supports hierarchy widget."""
        mock_gitlab_version.get.return_value = "18.7.1"
        graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
        gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

        tool = GetWorkItem(description="get work item", metadata=metadata)
        await tool._arun(project_id="namespace/project", work_item_iid=42)

        call_args = gitlab_client_mock.graphql.call_args[0]
        variables = call_args[1]
        assert variables["includeHierarchyWidget"] is True


@pytest.fixture(name="work_item_type_data")
def work_item_type_data_fixture():
    """Fixture for work item type data used by CreateWorkItem resolution."""
    return {
        "namespace": {
            "workItemTypes": {
                "nodes": [
                    {
                        "id": "gid://gitlab/WorkItems::Type/1",
                        "name": "Issue",
                    },
                ]
            }
        }
    }


@pytest.fixture(name="resolved_work_item")
def resolved_work_item_fixture(work_item_data):
    return ResolvedWorkItem(
        id="gid://gitlab/WorkItem/123",
        full_data={**work_item_data, "workItemType": {"name": "Issue"}},
        parent=ResolvedParent(type="project", full_path="namespace/project"),
    )


class TestCreateWorkItemAgentPlanVersionCompatibility:
    """Tests for CreateWorkItem version compatibility around AgentPlan."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_create_work_item_passes_agent_plan_on_new_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_type_data
    ):
        """On 19.0.0+, agentPlanWidget is included in the mutation input."""
        mock_gitlab_version.get.return_value = "19.0.0"
        gitlab_client_mock.graphql = AsyncMock(
            side_effect=[
                work_item_type_data,
                {
                    "workItemCreate": {
                        "workItem": {
                            "id": "gid://gitlab/WorkItem/1",
                            "title": "New Work Item",
                        },
                        "errors": [],
                    }
                },
            ]
        )

        tool = CreateWorkItem(description="create work item", metadata=metadata)
        await tool._arun(
            group_id="namespace/group",
            title="New Work Item",
            type_name="Issue",
            agent_plan="## Why\n\nReason",
        )

        mutation_call = gitlab_client_mock.graphql.call_args_list[1][0]
        variables = mutation_call[1]
        assert variables["input"]["agentPlanWidget"] == {"content": "## Why\n\nReason"}

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_create_work_item_drops_agent_plan_on_old_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, work_item_type_data
    ):
        """On <19.0.0, agentPlanWidget is dropped from the input and a warning is returned."""
        mock_gitlab_version.get.return_value = "18.11.0"
        gitlab_client_mock.graphql = AsyncMock(
            side_effect=[
                work_item_type_data,
                {
                    "workItemCreate": {
                        "workItem": {
                            "id": "gid://gitlab/WorkItem/1",
                            "title": "New Work Item",
                        },
                        "errors": [],
                    }
                },
            ]
        )

        tool = CreateWorkItem(description="create work item", metadata=metadata)
        result = await tool._arun(
            group_id="namespace/group",
            title="New Work Item",
            type_name="Issue",
            agent_plan="## Why\n\nReason",
        )

        mutation_call = gitlab_client_mock.graphql.call_args_list[1][0]
        variables = mutation_call[1]
        assert "agentPlanWidget" not in variables["input"]

        warnings = json.loads(result)["warnings"]
        assert any("agent_plan" in w for w in warnings)


class TestUpdateWorkItemAgentPlanVersionCompatibility:
    """Tests for UpdateWorkItem version compatibility around AgentPlan."""

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_update_work_item_passes_agent_plan_on_new_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, resolved_work_item
    ):
        """On 19.0.0+, agentPlanWidget is included in the mutation input."""
        mock_gitlab_version.get.return_value = "19.0.0"
        gitlab_client_mock.graphql = AsyncMock(
            return_value={
                "data": {
                    "workItemUpdate": {
                        "workItem": {
                            "id": "gid://gitlab/WorkItem/123",
                            "title": "Updated",
                            "state": "opened",
                        }
                    }
                }
            }
        )

        tool = UpdateWorkItem(description="update", metadata=metadata)
        tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)

        await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            agent_plan="## Why\n\nReason",
        )

        mutation, variables = gitlab_client_mock.graphql.call_args[0]
        assert "workItemUpdate" in mutation
        assert variables["input"]["agentPlanWidget"] == {"content": "## Why\n\nReason"}

    @pytest.mark.asyncio
    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    async def test_update_work_item_drops_agent_plan_on_old_version(
        self, mock_gitlab_version, gitlab_client_mock, metadata, resolved_work_item
    ):
        """On <19.0.0, agentPlanWidget is dropped from the input and a warning is returned."""
        mock_gitlab_version.get.return_value = "18.10.0"
        gitlab_client_mock.graphql = AsyncMock(
            return_value={
                "data": {
                    "workItemUpdate": {
                        "workItem": {
                            "id": "gid://gitlab/WorkItem/123",
                            "title": "Updated",
                            "state": "opened",
                        }
                    }
                }
            }
        )

        tool = UpdateWorkItem(description="update", metadata=metadata)
        tool._resolve_work_item_data = AsyncMock(return_value=resolved_work_item)

        result = await tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            agent_plan="## Why\n\nReason",
        )

        _, variables = gitlab_client_mock.graphql.call_args[0]
        assert "agentPlanWidget" not in variables["input"]

        warnings = json.loads(result)["warnings"]
        assert any("agent_plan" in w for w in warnings)
