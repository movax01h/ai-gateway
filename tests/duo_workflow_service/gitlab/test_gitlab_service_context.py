import asyncio
from unittest.mock import Mock

import pytest

from duo_workflow_service.gitlab.gitlab_instance_info_service import (
    GitLabInstanceInfo,
    GitLabInstanceInfoService,
)
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext


class TestGitLabServiceContext:
    """Test cases for GitLabServiceContext."""

    def test_context_manager_basic_usage(self):
        """Test basic context manager functionality."""
        # Arrange
        mock_service = Mock(spec=GitLabInstanceInfoService)
        expected_info = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0",
        )
        mock_service.create_from_project_and_namespace.return_value = expected_info

        project = {"web_url": "https://gitlab.com/test/project"}
        namespace = {"web_url": "https://gitlab.com/test"}

        # Act & Assert
        # Outside context, should be None
        assert GitLabServiceContext.get_current_instance_info() is None

        with GitLabServiceContext(mock_service, project, namespace):
            # Inside context, should have the info
            actual_info = GitLabServiceContext.get_current_instance_info()
            assert actual_info == expected_info

        # Outside context again, should be None
        assert GitLabServiceContext.get_current_instance_info() is None

        # Verify service was called correctly
        mock_service.create_from_project_and_namespace.assert_called_once_with(
            project, namespace
        )

    def test_context_manager_with_none_values(self):
        """Test context manager with None project and namespace."""
        # Arrange
        mock_service = Mock(spec=GitLabInstanceInfoService)
        fallback_info = GitLabInstanceInfo(
            instance_type="Unknown", instance_url="Unknown", instance_version="Unknown"
        )
        mock_service.create_from_project_and_namespace.return_value = fallback_info

        # Act & Assert
        with GitLabServiceContext(mock_service, None, None):
            actual_info = GitLabServiceContext.get_current_instance_info()
            assert actual_info == fallback_info

        mock_service.create_from_project_and_namespace.assert_called_once_with(
            None, None
        )

    def test_context_manager_nested_contexts(self):
        """Test nested context managers work correctly."""
        # Arrange
        mock_service1 = Mock(spec=GitLabInstanceInfoService)
        mock_service2 = Mock(spec=GitLabInstanceInfoService)

        info1 = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0",
        )
        info2 = GitLabInstanceInfo(
            instance_type="Self-Managed",
            instance_url="https://gitlab.example.com",
            instance_version="16.4.0",
        )

        mock_service1.create_from_project_and_namespace.return_value = info1
        mock_service2.create_from_project_and_namespace.return_value = info2

        project1 = {"web_url": "https://gitlab.com/test1"}
        project2 = {"web_url": "https://gitlab.example.com/test2"}

        # Act & Assert
        assert GitLabServiceContext.get_current_instance_info() is None

        with GitLabServiceContext(mock_service1, project1, None):
            assert GitLabServiceContext.get_current_instance_info() == info1

            with GitLabServiceContext(mock_service2, project2, None):
                # Inner context should override
                assert GitLabServiceContext.get_current_instance_info() == info2

            # Back to outer context
            assert GitLabServiceContext.get_current_instance_info() == info1

        # Outside all contexts
        assert GitLabServiceContext.get_current_instance_info() is None

    def test_context_manager_exception_handling(self):
        """Test context manager properly cleans up even when exceptions occur."""
        # Arrange
        mock_service = Mock(spec=GitLabInstanceInfoService)
        test_info = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0",
        )
        mock_service.create_from_project_and_namespace.return_value = test_info

        # Act & Assert
        assert GitLabServiceContext.get_current_instance_info() is None

        with pytest.raises(ValueError, match="Test exception"):
            with GitLabServiceContext(mock_service, None, None):
                assert GitLabServiceContext.get_current_instance_info() == test_info
                raise ValueError("Test exception")

        # Context should be cleaned up even after exception
        assert GitLabServiceContext.get_current_instance_info() is None

    def test_context_manager_project_only(self):
        """Test context manager with only project data."""
        # Arrange
        mock_service = Mock(spec=GitLabInstanceInfoService)
        expected_info = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0",
        )
        mock_service.create_from_project_and_namespace.return_value = expected_info

        project = {"web_url": "https://gitlab.com/test/project"}

        # Act & Assert
        with GitLabServiceContext(mock_service, project=project):
            actual_info = GitLabServiceContext.get_current_instance_info()
            assert actual_info == expected_info

        mock_service.create_from_project_and_namespace.assert_called_once_with(
            project, None
        )

    def test_context_manager_namespace_only(self):
        """Test context manager with only namespace data."""
        # Arrange
        mock_service = Mock(spec=GitLabInstanceInfoService)
        expected_info = GitLabInstanceInfo(
            instance_type="Self-Managed",
            instance_url="https://gitlab.example.com",
            instance_version="16.4.0",
        )
        mock_service.create_from_project_and_namespace.return_value = expected_info

        namespace = {"web_url": "https://gitlab.example.com/test"}

        # Act & Assert
        with GitLabServiceContext(mock_service, namespace=namespace):
            actual_info = GitLabServiceContext.get_current_instance_info()
            assert actual_info == expected_info

        mock_service.create_from_project_and_namespace.assert_called_once_with(
            None, namespace
        )

    @pytest.mark.asyncio
    async def test_context_manager_async_isolation(self):
        """Test that context is properly isolated between async tasks."""
        # Arrange
        mock_service1 = Mock(spec=GitLabInstanceInfoService)
        mock_service2 = Mock(spec=GitLabInstanceInfoService)

        info1 = GitLabInstanceInfo(
            instance_type="GitLab.com (SaaS)",
            instance_url="https://gitlab.com",
            instance_version="16.5.0",
        )
        info2 = GitLabInstanceInfo(
            instance_type="Self-Managed",
            instance_url="https://gitlab.example.com",
            instance_version="16.4.0",
        )

        mock_service1.create_from_project_and_namespace.return_value = info1
        mock_service2.create_from_project_and_namespace.return_value = info2

        results = []

        async def task1():
            with GitLabServiceContext(
                mock_service1, {"web_url": "https://gitlab.com/test1"}, None
            ):
                await asyncio.sleep(0.01)  # Yield control
                info = GitLabServiceContext.get_current_instance_info()
                results.append(("task1", info))

        async def task2():
            with GitLabServiceContext(
                mock_service2, {"web_url": "https://gitlab.example.com/test2"}, None
            ):
                await asyncio.sleep(0.01)  # Yield control
                info = GitLabServiceContext.get_current_instance_info()
                results.append(("task2", info))

        # Act
        await asyncio.gather(task1(), task2())

        # Assert
        assert len(results) == 2
        task1_result = next(r for r in results if r[0] == "task1")
        task2_result = next(r for r in results if r[0] == "task2")

        assert task1_result[1] == info1
        assert task2_result[1] == info2

    def test_get_current_instance_info_outside_context(self):
        """Test that get_current_instance_info returns None when not in context."""
        # Act & Assert
        assert GitLabServiceContext.get_current_instance_info() is None

    def test_context_manager_returns_self(self):
        """Test that context manager __enter__ returns self."""
        # Arrange
        mock_service = Mock(spec=GitLabInstanceInfoService)
        mock_service.create_from_project_and_namespace.return_value = (
            GitLabInstanceInfo(
                instance_type="Unknown",
                instance_url="Unknown",
                instance_version="Unknown",
            )
        )

        # Act & Assert
        with GitLabServiceContext(mock_service) as context:
            assert isinstance(context, GitLabServiceContext)
            assert context.service == mock_service
