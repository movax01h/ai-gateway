from typing import Optional
from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import (
    GitLabInstanceInfoService,
)


class TestGitLabInstanceInfoService:
    """Test cases for GitLabInstanceInfoService."""

    @pytest.fixture
    def project_gitlab_com(self) -> Project:
        """Sample GitLab.com project."""
        return Project(
            id=123,
            name="test-project",
            description="Test project",
            http_url_to_repo="https://gitlab.com/test/project.git",
            web_url="https://gitlab.com/test/project",
            default_branch="main",
            languages=[],
            exclusion_rules=[],
        )

    @pytest.fixture
    def project_self_managed(self) -> Project:
        """Sample self-managed GitLab project."""
        return Project(
            id=456,
            name="self-managed-project",
            description="Self-managed project",
            http_url_to_repo="https://gitlab.example.com/test/project.git",
            web_url="https://gitlab.example.com/test/project",
            default_branch="main",
            languages=[],
            exclusion_rules=[],
        )

    @pytest.fixture
    def project_dedicated(self) -> Project:
        """Sample GitLab Dedicated project."""
        return Project(
            id=789,
            name="dedicated-project",
            description="Dedicated project",
            http_url_to_repo="https://dedicated-example.gitlab.com/test/project.git",
            web_url="https://dedicated-example.gitlab.com/test/project",
            default_branch="main",
            languages=[],
            exclusion_rules=[],
        )

    @pytest.fixture
    def namespace_gitlab_com(self) -> Namespace:
        """Sample GitLab.com namespace."""
        return Namespace(
            id=123,
            name="test-namespace",
            description="Test namespace",
            web_url="https://gitlab.com/test-namespace",
        )

    @pytest.fixture
    def namespace_self_managed(self) -> Namespace:
        """Sample self-managed GitLab namespace."""
        return Namespace(
            id=456,
            name="self-managed-namespace",
            description="Self-managed namespace",
            web_url="https://gitlab.example.com/test-namespace",
        )

    @pytest.mark.parametrize(
        "project_fixture,mock_version,expected_type,expected_url,expected_version",
        [
            (
                "project_gitlab_com",
                "16.5.0-ee",
                "GitLab.com (SaaS)",
                "https://gitlab.com",
                "16.5.0-ee",
            ),
            (
                "project_self_managed",
                "15.0.0-ee",
                "Self-Managed",
                "https://gitlab.example.com",
                "15.0.0-ee",
            ),
            (
                "project_dedicated",
                "16.0.0-ee",
                "GitLab Dedicated",
                "https://dedicated-example.gitlab.com",
                "16.0.0-ee",
            ),
        ],
    )
    def test_create_from_project(
        self,
        request,
        project_fixture,
        mock_version,
        expected_type,
        expected_url,
        expected_version,
    ):
        """Test creating GitLab instance info from different project types."""
        project = request.getfixturevalue(project_fixture)

        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_gitlab_version:
            mock_gitlab_version.get.return_value = mock_version

            service = GitLabInstanceInfoService()
            result = service.create_from_project(project)

            assert result.instance_type == expected_type
            assert result.instance_url == expected_url
            assert result.instance_version == expected_version

    @pytest.mark.parametrize(
        "namespace_fixture,mock_version,expected_type,expected_url,expected_version",
        [
            (
                "namespace_gitlab_com",
                "16.5.0-ee",
                "GitLab.com (SaaS)",
                "https://gitlab.com",
                "16.5.0-ee",
            ),
            (
                "namespace_self_managed",
                "15.0.0-ee",
                "Self-Managed",
                "https://gitlab.example.com",
                "15.0.0-ee",
            ),
        ],
    )
    def test_create_from_namespace(
        self,
        request,
        namespace_fixture,
        mock_version,
        expected_type,
        expected_url,
        expected_version,
    ):
        """Test creating GitLab instance info from different namespace types."""
        namespace = request.getfixturevalue(namespace_fixture)

        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_gitlab_version:
            mock_gitlab_version.get.return_value = mock_version

            service = GitLabInstanceInfoService()
            result = service.create_from_namespace(namespace)

            assert result.instance_type == expected_type
            assert result.instance_url == expected_url
            assert result.instance_version == expected_version

    @pytest.mark.parametrize(
        "project_fixture,mock_version_return,mock_version_side_effect,expected_type,expected_url,expected_version",
        [
            # Version fallback - test with one instance type since logic is the same
            (
                "project_gitlab_com",
                None,
                None,
                "GitLab.com (SaaS)",
                "https://gitlab.com",
                "Unknown",
            ),
            # Version exception - test with one instance type since logic is the same
            (
                "project_gitlab_com",
                None,
                Exception("Version not available"),
                "GitLab.com (SaaS)",
                "https://gitlab.com",
                "Unknown",
            ),
        ],
    )
    def test_create_from_project_version_edge_cases(
        self,
        request,
        project_fixture,
        mock_version_return,
        mock_version_side_effect,
        expected_type,
        expected_url,
        expected_version,
    ):
        """Test creating GitLab instance info with version fallback and exception scenarios."""
        project = request.getfixturevalue(project_fixture)

        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_version:
            if mock_version_side_effect:
                mock_version.get.side_effect = mock_version_side_effect
            else:
                mock_version.get.return_value = mock_version_return

            service = GitLabInstanceInfoService()
            result = service.create_from_project(project)

            assert result.instance_type == expected_type
            assert result.instance_url == expected_url
            assert result.instance_version == expected_version

    @pytest.mark.parametrize(
        "input_fixture,method_name,expected_type,expected_url,expected_version",
        [
            # None project scenarios
            (None, "create_from_project", "Unknown", "Unknown", "Unknown"),
            # None namespace scenarios
            (None, "create_from_namespace", "Unknown", "Unknown", "Unknown"),
        ],
    )
    def test_create_from_none_inputs(
        self, input_fixture, method_name, expected_type, expected_url, expected_version
    ):
        """Test creating GitLab instance info when project or namespace is None."""
        service = GitLabInstanceInfoService()
        method = getattr(service, method_name)
        result = method(input_fixture)

        assert result.instance_type == expected_type
        assert result.instance_url == expected_url
        assert result.instance_version == expected_version

    @pytest.mark.parametrize(
        "project_fixture,namespace_fixture,mock_version,expected_type,expected_url,expected_version",
        [
            # Project priority over namespace - if this is even a possible scenario
            (
                "project_gitlab_com",
                "namespace_self_managed",
                "16.5.0-ee",
                "GitLab.com (SaaS)",
                "https://gitlab.com",
                "16.5.0-ee",
            ),
            # Fallback to namespace when no project
            (
                None,
                "namespace_self_managed",
                "15.0.0-ee",
                "Self-Managed",
                "https://gitlab.example.com",
                "15.0.0-ee",
            ),
            # Both None scenario
            (
                None,
                None,
                "16.5.0-ee",
                "Unknown",
                "Unknown",
                "Unknown",
            ),
        ],
    )
    def test_create_from_project_and_namespace(
        self,
        request,
        project_fixture,
        namespace_fixture,
        mock_version,
        expected_type,
        expected_url,
        expected_version,
    ):
        """Test creating GitLab instance info from project and namespace combinations."""
        project = request.getfixturevalue(project_fixture) if project_fixture else None
        namespace = (
            request.getfixturevalue(namespace_fixture) if namespace_fixture else None
        )

        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_gitlab_version:
            mock_gitlab_version.get.return_value = mock_version

            service = GitLabInstanceInfoService()
            result = service.create_from_project_and_namespace(project, namespace)

            assert result.instance_type == expected_type
            assert result.instance_url == expected_url
            assert result.instance_version == expected_version

    @pytest.mark.parametrize(
        "web_url,expected_type",
        [
            ("https://gitlab.com/test/project", "GitLab.com (SaaS)"),
            ("http://gitlab.com/test/project", "GitLab.com (SaaS)"),
            ("https://gitlab.com/", "GitLab.com (SaaS)"),
            ("https://dedicated-example.gitlab.com/test", "GitLab Dedicated"),
            ("https://dedicated-test.gitlab.com/", "GitLab Dedicated"),
            ("https://gitlab.example.com/test", "Self-Managed"),
            ("https://git.company.com/test", "Self-Managed"),
            ("http://192.168.1.100:8080/test", "Self-Managed"),
            ("", "Unknown"),
            ("Unknown", "Unknown"),
            # Edge case: project name contains "dedicated-" but it's on gitlab.com
            ("https://gitlab.com/dedicated-project/smoke-tests", "GitLab.com (SaaS)"),
            ("https://gitlab.com/org/dedicated-something", "GitLab.com (SaaS)"),
            ("https://gitlab.com/dedicated-team/dedicated-repo", "GitLab.com (SaaS)"),
            # Additional edge cases for the regex
            ("https://gitlab.com/user/project-dedicated-name", "GitLab.com (SaaS)"),
            (
                "https://gitlab.com/dedicated-",
                "GitLab.com (SaaS)",
            ),  # Edge case with trailing dash
            (
                "https://dedicated.gitlab.com/test",
                "GitLab.com (SaaS)",
            ),  # Missing dash after dedicated
            (
                "https://not-dedicated-example.gitlab.com/test",
                "GitLab.com (SaaS)",
            ),  # Doesn't start with dedicated-
        ],
    )
    def test_determine_instance_type_from_url(self, web_url, expected_type):
        """Test instance type determination from various URLs."""
        service = GitLabInstanceInfoService()
        result = service._determine_instance_type_from_url(web_url)
        assert result == expected_type

    @pytest.mark.parametrize(
        "web_url,expected_url",
        [
            ("https://gitlab.com/test/project", "https://gitlab.com"),
            ("http://gitlab.com/test/project", "http://gitlab.com"),
            ("https://gitlab.example.com/test/project", "https://gitlab.example.com"),
            ("https://git.company.com:8080/test", "https://git.company.com:8080"),
            ("http://192.168.1.100:8080/test", "http://192.168.1.100:8080"),
            ("", "Unknown"),
            ("Unknown", "Unknown"),
        ],
    )
    def test_extract_base_url_from_web_url(self, web_url, expected_url):
        """Test base URL extraction from various web URLs."""
        service = GitLabInstanceInfoService()
        result = service._extract_base_url_from_web_url(web_url)
        assert result == expected_url

    def test_get_gitlab_version_success(self):
        """Test successful GitLab version retrieval."""
        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_version:
            mock_version.get.return_value = "16.5.0-ee"

            service = GitLabInstanceInfoService()
            result = service._get_gitlab_version()
            assert result == "16.5.0-ee"

    def test_get_gitlab_version_none(self):
        """Test GitLab version retrieval when version is None."""
        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_version:
            mock_version.get.return_value = None

            service = GitLabInstanceInfoService()
            result = service._get_gitlab_version()
            assert result == "Unknown"

    def test_get_gitlab_version_exception(self):
        """Test GitLab version retrieval when exception is raised."""
        with patch(
            "duo_workflow_service.gitlab.gitlab_instance_info_service.gitlab_version"
        ) as mock_version:
            mock_version.get.side_effect = Exception("Version not available")

            service = GitLabInstanceInfoService()
            result = service._get_gitlab_version()
            assert result == "Unknown"
