# pylint: disable=file-naming-for-tests
"""Tests for version compatibility utilities."""
from unittest.mock import patch

import pytest
from packaging.version import Version

from duo_workflow_service.tools.work_items.version_compatibility import (
    BASE_DISCUSSION_ID_FIELD_VERSION,
    DEFAULT_FALLBACK_VERSION,
    HIERARCHY_WIDGET_VERSION,
    NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION,
    get_gitlab_version,
    get_query_variables_for_version,
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


class TestSupportsDiscussionIDField:
    """Tests for supports_discussion_id_field function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("18.9.0", True),  # Exact threshold version
            ("18.10.0", True),  # Above threshold (minor)
            ("18.9.1", True),  # Above threshold (patch)
            ("18.8.0", False),  # Below threshold
        ],
    )
    def test_supports_discussion_id_field(self, mock_get_version, version, expected):
        """Test discussion ID field support with various versions."""
        mock_get_version.return_value = Version(version)

        result = supports_discussion_id_field()

        assert result is expected


class TestSupportsNoteResolvedAndResolvableFields:
    """Tests for supports_note_resolved_and_resolvable_fields function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("18.9.0", True),  # Exact threshold version
            ("18.10.0", True),  # Above threshold (minor)
            ("18.9.1", True),  # Above threshold (patch)
            ("18.8.0", False),  # Below threshold
        ],
    )
    def test_supports_note_resolved_and_resolvable_fields(
        self, mock_get_version, version, expected
    ):
        """Test note resolved and resolvable fields support with various versions."""
        mock_get_version.return_value = Version(version)

        result = supports_note_resolved_and_resolvable_fields()

        assert result is expected


class TestSupportsHierarchyWidget:
    """Tests for supports_hierarchy_widget function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.get_gitlab_version"
    )
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("18.7.0", True),  # Exact threshold version
            ("18.8.0", True),  # Above threshold (minor)
            ("18.7.1", True),  # Above threshold (patch)
            ("18.6.0", False),  # Below threshold (minor)
            ("17.0.0", False),  # Below threshold (major)
            ("18.6.9", False),  # Below threshold (patch)
        ],
    )
    def test_supports_hierarchy_widget(self, mock_get_version, version, expected):
        """Test hierarchy widget support with various versions."""
        mock_get_version.return_value = Version(version)

        result = supports_hierarchy_widget()

        assert result is expected


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
    @pytest.mark.parametrize(
        "hierarchy,note_resolved,discussion_id,expected",
        [
            (
                True,
                False,
                False,
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": False,
                },
            ),
            (
                False,
                True,
                False,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": True,
                    "includeDiscussionIdField": False,
                },
            ),
            (
                False,
                False,
                True,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": True,
                },
            ),
            (
                False,
                False,
                False,
                {
                    "includeHierarchyWidget": False,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": False,
                },
            ),
            (
                True,
                True,
                True,
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": True,
                    "includeDiscussionIdField": True,
                },
            ),
        ],
    )
    def test_get_query_variables_for_version(
        self,
        mock_supports_discussion_id,
        mock_supports_note_resolved,
        mock_supports_hierarchy,
        hierarchy,
        note_resolved,
        discussion_id,
        expected,
    ):
        """Test query variables for different feature support combinations."""
        mock_supports_hierarchy.return_value = hierarchy
        mock_supports_note_resolved.return_value = note_resolved
        mock_supports_discussion_id.return_value = discussion_id

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
    @pytest.mark.parametrize(
        "requested_keys,expected",
        [
            (
                (),
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": True,
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
                (
                    "includeHierarchyWidget",
                    "includeNoteResolvedAndResolvableFields",
                    "includeDiscussionIdField",
                ),
                {
                    "includeHierarchyWidget": True,
                    "includeNoteResolvedAndResolvableFields": False,
                    "includeDiscussionIdField": True,
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

    def test_default_fallback_version_constant(self):
        """Test that DEFAULT_FALLBACK_VERSION is set correctly."""
        assert DEFAULT_FALLBACK_VERSION == Version("18.6.0")

    def test_fallback_version_is_below_hierarchy_threshold(self):
        """Test that fallback version is below hierarchy widget threshold."""
        assert DEFAULT_FALLBACK_VERSION < HIERARCHY_WIDGET_VERSION
