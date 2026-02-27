# pylint: disable=file-naming-for-tests
"""Tests for version compatibility utilities."""
from unittest.mock import patch

import pytest
from packaging.version import Version

from duo_workflow_service.tools.work_items.version_compatibility import (
    BASE_DISCUSSION_ID_FIELD_VERSION,
    DEFAULT_FALLBACK_VERSION,
    DEVELOPMENT_WIDGET_VERSION,
    HIERARCHY_WIDGET_VERSION,
    NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION,
    get_gitlab_version,
    get_query_variables_for_version,
    supports_development_widget,
    supports_discussion_id_field,
    supports_hierarchy_widget,
    supports_note_resolved_and_resolvable_fields,
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


class TestVersionCompatibilityFunctions:
    """Tests for version compatibility functions."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    @pytest.mark.parametrize(
        "compatibility_func,gitlab_version,expected_result",
        [
            # supports_discussion_id_field (threshold: 18.9.0)
            (supports_discussion_id_field, "18.9.0", True),
            (supports_discussion_id_field, "18.10.0", True),
            (supports_discussion_id_field, "18.9.1", True),
            (supports_discussion_id_field, "18.8.0", False),
            # supports_note_resolved_and_resolvable_fields (threshold: 18.9.0)
            (supports_note_resolved_and_resolvable_fields, "18.9.0", True),
            (supports_note_resolved_and_resolvable_fields, "18.10.0", True),
            (supports_note_resolved_and_resolvable_fields, "18.9.1", True),
            (supports_note_resolved_and_resolvable_fields, "18.8.0", False),
            # supports_development_widget (threshold: 18.9.0)
            (supports_development_widget, "18.9.0", True),
            (supports_development_widget, "18.10.0", True),
            (supports_development_widget, "18.9.1", True),
            (supports_development_widget, "18.8.0", False),
            # supports_hierarchy_widget (threshold: 18.7.0)
            (supports_hierarchy_widget, "18.7.0", True),
            (supports_hierarchy_widget, "18.8.0", True),
            (supports_hierarchy_widget, "18.7.1", True),
            (supports_hierarchy_widget, "18.6.0", False),
            (supports_hierarchy_widget, "17.0.0", False),
            (supports_hierarchy_widget, "18.6.9", False),
        ],
    )
    def test_version_compatibility(
        self, mock_get_version, compatibility_func, gitlab_version, expected_result
    ):
        """Test version compatibility functions with various versions."""
        mock_get_version.return_value = Version(gitlab_version)
        assert compatibility_func() is expected_result


class TestGetQueryVariablesForVersion:
    """Tests for get_query_variables_for_version function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_hierarchy_widget"
    )
    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_note_resolved_and_resolvable_fields"
    )
    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_discussion_id_field"
    )
    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_development_widget"
    )
    @pytest.mark.parametrize(
        "hierarchy,note_resolved,discussion_id,development,expected",
        [
            (
                True,
                False,
                False,
                False,
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": False,
                    "includeDevelopmentWidget": False,
                },
            ),
            (
                False,
                True,
                False,
                False,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": True,
                    "includeDiscussionIdField": False,
                    "includeDevelopmentWidget": False,
                },
            ),
            (
                False,
                False,
                True,
                False,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": True,
                    "includeDevelopmentWidget": False,
                },
            ),
            (
                False,
                False,
                False,
                True,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": False,
                    "includeDevelopmentWidget": True,
                },
            ),
            (
                False,
                False,
                False,
                False,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": False,
                    "includeDevelopmentWidget": False,
                },
            ),
            (
                True,
                True,
                True,
                True,
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": True,
                    "includeDiscussionIdField": True,
                    "includeDevelopmentWidget": True,
                },
            ),
        ],
    )
    def test_get_query_variables_for_version(
        self,
        mock_supports_development,
        mock_supports_discussion_id,
        mock_supports_note_resolved,
        mock_supports_hierarchy,
        hierarchy,
        note_resolved,
        discussion_id,
        development,
        expected,
    ):
        """Test query variables for different feature support combinations."""
        mock_supports_hierarchy.return_value = hierarchy
        mock_supports_note_resolved.return_value = note_resolved
        mock_supports_discussion_id.return_value = discussion_id
        mock_supports_development.return_value = development

        result = get_query_variables_for_version()

        assert result == expected

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_hierarchy_widget"
    )
    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_note_resolved_and_resolvable_fields"
    )
    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_discussion_id_field"
    )
    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_development_widget"
    )
    @pytest.mark.parametrize(
        "requested_keys,expected",
        [
            (
                (),
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": True,
                    "includeDevelopmentWidget": False,
                },
            ),
            (
                ("includeHierarchyWidget",),
                {"includeHierarchyWidget": True},
            ),
            (
                ("includeHierarchyWidget", "includeDiscussionIdField"),
                {
                    "includeHierarchyWidget": True,
                    "includeDiscussionIdField": True,
                },
            ),
            (
                ("includeNoteResolvedAndResolvableFields",),
                {"includeNoteResolvedAndResolvableFields": False},
            ),
            (
                ("includeDiscussionIdField",),
                {"includeDiscussionIdField": True},
            ),
            (
                ("includeDevelopmentWidget",),
                {"includeDevelopmentWidget": False},
            ),
            (
                (
                    "includeHierarchyWidget",
                    "includeNoteResolvedAndResolvableFields",
                    "includeDiscussionIdField",
                    "includeDevelopmentWidget",
                ),
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": True,
                    "includeDevelopmentWidget": False,
                },
            ),
            (
                ("includeHierarchyWidget", "invalidKey"),
                {"includeHierarchyWidget": True},
            ),
            (
                ("invalidKey"),
                {},
            ),
        ],
    )
    def test_get_query_variables_for_version_with_requested_keys(
        self,
        mock_supports_development,
        mock_supports_discussion_id,
        mock_supports_note_resolved,
        mock_supports_hierarchy,
        requested_keys,
        expected,
    ):
        """Test query variables with specific requested keys."""
        mock_supports_hierarchy.return_value = True
        mock_supports_note_resolved.return_value = False
        mock_supports_discussion_id.return_value = True
        mock_supports_development.return_value = False

        result = get_query_variables_for_version(*requested_keys)

        assert result == expected


class TestVersionConstants:
    """Tests for version constants."""

    def test_hierarchy_widget_version_constant(self):
        """Test that HIERARCHY_WIDGET_VERSION is set correctly."""
        assert HIERARCHY_WIDGET_VERSION == Version("18.7.0")

    def test_note_resolvable_and_resolved_version_constant(self):
        """Test that NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION is set correctly."""
        assert NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION == Version("18.9.0")

    def test_discussion_id_version_constant(self):
        """Test that BASE_DISCUSSION_ID_FIELD_VERSION is set correctly."""
        assert BASE_DISCUSSION_ID_FIELD_VERSION == Version("18.9.0")

    def test_development_widget_version_constant(self):
        """Test that DEVELOPMENT_WIDGET_VERSION is set correctly."""
        assert DEVELOPMENT_WIDGET_VERSION == Version("18.9.0")

    def test_default_fallback_version_constant(self):
        """Test that DEFAULT_FALLBACK_VERSION is set correctly."""
        assert DEFAULT_FALLBACK_VERSION == Version("18.6.0")

    def test_fallback_version_is_below_hierarchy_threshold(self):
        """Test that fallback version is below hierarchy widget threshold."""
        assert DEFAULT_FALLBACK_VERSION < HIERARCHY_WIDGET_VERSION
