"""Tests for version compatibility utilities."""

from unittest.mock import patch

import pytest

from duo_workflow_service.tools.work_items.version_compatibility import (
    get_query_variables_for_version,
    get_query_with_agent_plan_widget,
)

_PLACEHOLDER = "# AGENT_PLAN_WIDGET_PLACEHOLDER"
_FRAGMENT = (
    "... on WorkItemWidgetAgentPlan {\n                    content\n                }"
)


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


class TestGetQueryWithAgentPlanWidget:
    """Tests for get_query_with_agent_plan_widget function."""

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_agent_plan_widget",
        return_value=True,
    )
    def test_injects_fragment_when_supported(self, _mock):
        """Fragment replaces placeholder when GitLab >= 19.0."""
        base = f"widgets {{\n    {_PLACEHOLDER}\n}}"
        result = get_query_with_agent_plan_widget(base)
        assert _FRAGMENT in result
        assert _PLACEHOLDER not in result

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_agent_plan_widget",
        return_value=False,
    )
    def test_removes_placeholder_when_unsupported(self, _mock):
        """Placeholder is removed (not replaced with fragment) on GitLab < 19.0."""
        base = f"widgets {{\n    {_PLACEHOLDER}\n}}"
        result = get_query_with_agent_plan_widget(base)
        assert _FRAGMENT not in result
        assert _PLACEHOLDER not in result

    @patch(
        "duo_workflow_service.tools.work_items.version_compatibility.supports_agent_plan_widget",
        return_value=False,
    )
    def test_query_is_valid_without_fragment(self, _mock):
        """Resulting query on older GitLab must contain no unknown type reference."""
        base = f"mutation foo {{\n    widgets {{\n        {_PLACEHOLDER}\n    }}\n}}"
        result = get_query_with_agent_plan_widget(base)
        assert "WorkItemWidgetAgentPlan" not in result
