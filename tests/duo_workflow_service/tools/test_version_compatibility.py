"""Tests for version compatibility utilities."""

from unittest.mock import patch

import pytest
from packaging.version import Version

from duo_workflow_service.tools.version_compatibility import (
    AGENT_PLAN_WIDGET_VERSION,
    BASE_DISCUSSION_ID_FIELD_VERSION,
    DEFAULT_FALLBACK_VERSION,
    DEVELOPMENT_WIDGET_VERSION,
    GROUP_LEVEL_CUSTOM_INSTRUCTIONS_VERSION,
    HIERARCHY_WIDGET_VERSION,
    LICENSED_FEATURE_AVAILABILITY_VERSION,
    NOTE_RESOLVABLE_AND_RESOLVED_FIELDS_VERSION,
    get_gitlab_version,
    supports_agent_plan_widget,
    supports_development_widget,
    supports_discussion_id_field,
    supports_group_level_custom_instructions,
    supports_hierarchy_widget,
    supports_licensed_feature_availability,
    supports_note_resolved_and_resolvable_fields,
)


class TestGetGitLabVersion:
    """Tests for get_gitlab_version function."""

    @patch("duo_workflow_service.tools.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_valid_version(self, mock_gitlab_version):
        """Test getting GitLab version when valid version is available."""
        mock_gitlab_version.get.return_value = "18.7.0"

        result = get_gitlab_version()

        assert result == Version("18.7.0")
        mock_gitlab_version.get.assert_called_once()

    @patch("duo_workflow_service.tools.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_patch_version(self, mock_gitlab_version):
        """Test getting GitLab version with patch version."""
        mock_gitlab_version.get.return_value = "18.7.1"

        result = get_gitlab_version()

        assert result == Version("18.7.1")

    @patch("duo_workflow_service.tools.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_older_version(self, mock_gitlab_version):
        """Test getting GitLab version with older version."""
        mock_gitlab_version.get.return_value = "18.6.0"

        result = get_gitlab_version()

        assert result == Version("18.6.0")

    @patch("duo_workflow_service.tools.version_compatibility.gitlab_version")
    def test_get_gitlab_version_when_none(self, mock_gitlab_version):
        """Test fallback when version is None."""
        mock_gitlab_version.get.return_value = None

        result = get_gitlab_version()

        assert result == DEFAULT_FALLBACK_VERSION
        mock_gitlab_version.get.assert_called_once()

    @patch("duo_workflow_service.tools.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_invalid_version(self, mock_gitlab_version):
        """Test fallback when version string is invalid."""
        mock_gitlab_version.get.return_value = "invalid-version"

        result = get_gitlab_version()

        assert result == DEFAULT_FALLBACK_VERSION

    @patch("duo_workflow_service.tools.version_compatibility.gitlab_version")
    def test_get_gitlab_version_with_type_error(self, mock_gitlab_version):
        """Test fallback when TypeError is raised."""
        mock_gitlab_version.get.side_effect = TypeError("Invalid type")

        result = get_gitlab_version()

        assert result == DEFAULT_FALLBACK_VERSION


class TestVersionCompatibilityFunctions:
    """Tests for version compatibility functions."""

    @patch("duo_workflow_service.tools.version_compatibility.get_gitlab_version")
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
            # supports_licensed_feature_availability (threshold: 18.11.0)
            (supports_licensed_feature_availability, "18.11.0", True),
            (supports_licensed_feature_availability, "19.0.0", True),
            (supports_licensed_feature_availability, "18.10.0", False),
            # supports_group_level_custom_instructions (threshold: 19.0.0)
            (supports_group_level_custom_instructions, "19.0.0", True),
            (supports_group_level_custom_instructions, "19.1.0", True),
            (supports_group_level_custom_instructions, "20.0.0", True),
            (supports_group_level_custom_instructions, "18.11.0", False),
            (supports_group_level_custom_instructions, "18.0.0", False),
            # supports_agent_plan_widget (threshold: 19.0.0)
            (supports_agent_plan_widget, "19.0.0", True),
            (supports_agent_plan_widget, "19.1.0", True),
            (supports_agent_plan_widget, "19.0.1", True),
            (supports_agent_plan_widget, "18.11.0", False),
            (supports_agent_plan_widget, "18.6.0", False),
        ],
    )
    def test_version_compatibility(
        self, mock_get_version, compatibility_func, gitlab_version, expected_result
    ):
        """Test version compatibility functions with various versions."""
        mock_get_version.return_value = Version(gitlab_version)
        assert compatibility_func() is expected_result


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

    def test_licensed_feature_availability_version_constant(self):
        """Test that LICENSED_FEATURE_AVAILABILITY_VERSION is set correctly."""
        assert LICENSED_FEATURE_AVAILABILITY_VERSION == Version("18.11.0")

    def test_group_level_custom_instructions_version_constant(self):
        """Test that GROUP_LEVEL_CUSTOM_INSTRUCTIONS_VERSION is set correctly."""
        assert GROUP_LEVEL_CUSTOM_INSTRUCTIONS_VERSION == Version("19.0.0")

    def test_agent_plan_widget_version_constant(self):
        """Test that AGENT_PLAN_WIDGET_VERSION is set correctly."""
        assert AGENT_PLAN_WIDGET_VERSION == Version("19.0.0")

    def test_fallback_version_is_below_agent_plan_widget_threshold(self):
        """Test that fallback version is below agent plan widget threshold."""
        assert DEFAULT_FALLBACK_VERSION < AGENT_PLAN_WIDGET_VERSION
