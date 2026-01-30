"""Tests for tool supersedes functionality.

Ensures that tools superseding other tools have the same name as the superseded tool. This constraint is critical for
proper tool replacement in the tool registry.
"""

from duo_workflow_service.tools.utils import get_all_ops_tools


class TestToolSupersedes:
    """Test suite for tool supersedes constraints."""

    def test_superseding_tools_have_same_name_as_superseded_tool(self):
        """Verify superseding tools have the same name as superseded tools."""
        mismatches = []

        for tool_class in get_all_ops_tools():
            superseded_class = getattr(tool_class, "supersedes", None)
            if superseded_class:
                # Get name from Pydantic model_fields
                superseding_name = tool_class.model_fields["name"].default
                superseded_name = superseded_class.model_fields["name"].default

                if superseding_name != superseded_name:
                    mismatches.append(
                        f"{tool_class.__name__} (name='{superseding_name}') "
                        f"supersedes {superseded_class.__name__} (name='{superseded_name}')"
                    )

        assert (
            not mismatches
        ), f"Found {len(mismatches)} tool(s) with mismatched names:\n" + "\n".join(
            f"  - {m}" for m in mismatches
        )
