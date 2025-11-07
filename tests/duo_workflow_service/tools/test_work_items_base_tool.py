import pytest

from duo_workflow_service.tools.work_items.base_tool import WorkItemBaseTool


class TestBuildHierarchyWidget:
    """Test the _build_hierarchy_widget static method."""

    def test_build_hierarchy_widget_with_valid_gid(self):
        """Test building hierarchy widget with valid GitLab GID."""
        kwargs = {"hierarchy_widget": {"parent_id": "gid://gitlab/WorkItem/123"}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result == {"parentId": "gid://gitlab/WorkItem/123"}
        assert warnings == []

    def test_build_hierarchy_widget_with_no_hierarchy_widget(self):
        """Test building hierarchy widget when no hierarchy_widget is provided."""
        kwargs = {}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == []

    def test_build_hierarchy_widget_with_none_hierarchy_widget(self):
        """Test building hierarchy widget when hierarchy_widget is None."""
        kwargs = {"hierarchy_widget": None}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == []

    def test_build_hierarchy_widget_with_invalid_type(self):
        """Test building hierarchy widget with invalid type (not a dict)."""
        kwargs = {"hierarchy_widget": "invalid_string"}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == ["hierarchy_widget must be a dictionary"]

    def test_build_hierarchy_widget_with_missing_parent_id(self):
        """Test building hierarchy widget with missing parent_id key."""
        kwargs = {"hierarchy_widget": {"other_key": "value"}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == ["hierarchy_widget must contain 'parent_id' key"]

    def test_build_hierarchy_widget_with_empty_parent_id(self):
        """Test building hierarchy widget with empty parent_id."""
        kwargs = {"hierarchy_widget": {"parent_id": ""}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == ["hierarchy_widget must contain 'parent_id' key"]

    def test_build_hierarchy_widget_with_none_parent_id(self):
        """Test building hierarchy widget with None parent_id."""
        kwargs = {"hierarchy_widget": {"parent_id": None}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == ["hierarchy_widget must contain 'parent_id' key"]

    def test_build_hierarchy_widget_with_invalid_gid_format(self):
        """Test building hierarchy widget with invalid GID format."""
        kwargs = {"hierarchy_widget": {"parent_id": "invalid_gid_format"}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == [
            "Invalid parent_id format: invalid_gid_format. Expected GitLab GID."
        ]

    def test_build_hierarchy_widget_with_numeric_id(self):
        """Test building hierarchy widget with numeric ID (should fail)."""
        kwargs = {"hierarchy_widget": {"parent_id": "123"}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == ["Invalid parent_id format: 123. Expected GitLab GID."]

    def test_build_hierarchy_widget_with_partial_gid(self):
        """Test building hierarchy widget with partial GID (should fail)."""
        kwargs = {"hierarchy_widget": {"parent_id": "gid://gitlab/"}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == [
            "Invalid parent_id format: gid://gitlab/. Expected GitLab GID."
        ]

    def test_build_hierarchy_widget_camel_case_conversion(self):
        """Test that parent_id is converted to camelCase parentId for GraphQL."""
        kwargs = {"hierarchy_widget": {"parent_id": "gid://gitlab/WorkItem/789"}}
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result == {"parentId": "gid://gitlab/WorkItem/789"}
        assert "parent_id" not in result  # Ensure snake_case is not present
        assert warnings == []

    def test_build_hierarchy_widget_accumulates_warnings(self):
        """Test that warnings are accumulated properly."""
        kwargs = {"hierarchy_widget": "not_a_dict"}
        warnings = ["existing_warning"]

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result is None
        assert warnings == ["existing_warning", "hierarchy_widget must be a dictionary"]

    def test_build_hierarchy_widget_with_extra_keys(self):
        """Test building hierarchy widget with extra keys (should be ignored)."""
        kwargs = {
            "hierarchy_widget": {
                "parent_id": "gid://gitlab/WorkItem/123",
                "extra_key": "ignored_value",
            }
        }
        warnings = []

        result = WorkItemBaseTool._build_hierarchy_widget(kwargs, warnings)

        assert result == {"parentId": "gid://gitlab/WorkItem/123"}
        assert warnings == []
