# pylint: disable=file-naming-for-tests
"""Integration tests for version compatibility in work item tools."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from duo_workflow_service.tools.work_item import GetWorkItem, ListWorkItems


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
