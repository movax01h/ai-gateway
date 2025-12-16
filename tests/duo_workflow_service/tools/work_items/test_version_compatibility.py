# pylint: disable=file-naming-for-tests
"""Tests for version compatibility utilities."""
from unittest.mock import patch

from packaging.version import Version

from duo_workflow_service.tools.work_items.version_compatibility import (
    DEFAULT_FALLBACK_VERSION,
    HIERARCHY_WIDGET_VERSION,
    get_gitlab_version,
    get_query_variables_for_version,
    supports_hierarchy_widget,
)


class TestGetGitLabVersion:
    """Tests for get_gitlab_version function."""

    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_valid_version(self, mock_gitlab_version):
        """Test getting GitLab version when valid version is available."""
        mock_gitlab_version.get.return_value = "18.7.0"

        result = get_gitlab_version()

        assert result == Version("18.7.0")
        mock_gitlab_version.get.assert_called_once()

    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_patch_version(self, mock_gitlab_version):
        """Test getting GitLab version with patch version."""
        mock_gitlab_version.get.return_value = "18.7.1"

        result = get_gitlab_version()

        assert result == Version("18.7.1")

    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_older_version(self, mock_gitlab_version):
        """Test getting GitLab version with older version."""
        mock_gitlab_version.get.return_value = "18.6.0"

        result = get_gitlab_version()

        assert result == Version("18.6.0")

    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    def test_get_gitlab_version_when_none(self, mock_gitlab_version):
        """Test fallback when version is None."""
        mock_gitlab_version.get.return_value = None

        result = get_gitlab_version()

        assert result == DEFAULT_FALLBACK_VERSION
        mock_gitlab_version.get.assert_called_once()

    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_invalid_version(self, mock_gitlab_version):
        """Test fallback when version string is invalid."""
        mock_gitlab_version.get.return_value = "invalid-version"

        result = get_gitlab_version()

        assert result == DEFAULT_FALLBACK_VERSION

    @patch("duo_workflow_service.tools.work_items.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_type_error(self, mock_gitlab_version):
        """Test fallback when TypeError is raised."""
        mock_gitlab_version.get.side_effect = TypeError("Invalid type")

        result = get_gitlab_version()

        assert result == DEFAULT_FALLBACK_VERSION


class TestSupportsHierarchyWidget:
    """Tests for supports_hierarchy_widget function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    def test_supports_hierarchy_widget_with_exact_version(self, mock_get_version):
        """Test hierarchy widget support with exact threshold version."""
        mock_get_version.return_value = Version("18.7.0")

        result = supports_hierarchy_widget()

        assert result is True

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    def test_supports_hierarchy_widget_with_newer_version(self, mock_get_version):
        """Test hierarchy widget support with newer version."""
        mock_get_version.return_value = Version("18.8.0")

        result = supports_hierarchy_widget()

        assert result is True

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    def test_supports_hierarchy_widget_with_patch_version(self, mock_get_version):
        """Test hierarchy widget support with patch version above threshold."""
        mock_get_version.return_value = Version("18.7.1")

        result = supports_hierarchy_widget()

        assert result is True

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    def test_does_not_support_hierarchy_widget_with_older_version(
        self, mock_get_version
    ):
        """Test hierarchy widget not supported with older version."""
        mock_get_version.return_value = Version("18.6.0")

        result = supports_hierarchy_widget()

        assert result is False

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    def test_does_not_support_hierarchy_widget_with_much_older_version(
        self, mock_get_version
    ):
        """Test hierarchy widget not supported with much older version."""
        mock_get_version.return_value = Version("17.0.0")

        result = supports_hierarchy_widget()

        assert result is False

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    def test_does_not_support_hierarchy_widget_just_below_threshold(
        self, mock_get_version
    ):
        """Test hierarchy widget not supported just below threshold."""
        mock_get_version.return_value = Version("18.6.9")

        result = supports_hierarchy_widget()

        assert result is False


class TestGetQueryVariablesForVersion:
    """Tests for get_query_variables_for_version function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_hierarchy_widget"
    )
    def test_get_query_variables_when_hierarchy_supported(
        self, mock_supports_hierarchy
    ):
        """Test query variables when hierarchy widget is supported."""
        mock_supports_hierarchy.return_value = True

        result = get_query_variables_for_version()

        assert result == {"includeHierarchyWidget": True}

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_hierarchy_widget"
    )
    def test_get_query_variables_when_hierarchy_not_supported(
        self, mock_supports_hierarchy
    ):
        """Test query variables when hierarchy widget is not supported."""
        mock_supports_hierarchy.return_value = False

        result = get_query_variables_for_version()

        assert result == {"includeHierarchyWidget": False}


class TestVersionConstants:
    """Tests for version constants."""

    def test_hierarchy_widget_version_constant(self):
        """Test that HIERARCHY_WIDGET_VERSION is set correctly."""
        assert HIERARCHY_WIDGET_VERSION == Version("18.7.0")

    def test_default_fallback_version_constant(self):
        """Test that DEFAULT_FALLBACK_VERSION is set correctly."""
        assert DEFAULT_FALLBACK_VERSION == Version("18.6.0")

    def test_fallback_version_is_below_hierarchy_threshold(self):
        """Test that fallback version is below hierarchy widget threshold."""
        assert DEFAULT_FALLBACK_VERSION < HIERARCHY_WIDGET_VERSION
