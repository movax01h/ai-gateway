# pylint: disable=file-naming-for-tests
"""Test suite for UILogEventsSupervisor enum (v1)."""

from duo_workflow_service.agent_platform.v1.components.supervisor.ui_log import (
    UILogEventsSupervisor,
)
from duo_workflow_service.agent_platform.v1.ui_log import BaseUILogEvents


class TestUILogEventsSupervisor:
    """Tests for the UILogEventsSupervisor enum."""

    def test_inherits_from_base_ui_log_events(self):
        """Test that UILogEventsSupervisor inherits from BaseUILogEvents."""
        assert issubclass(UILogEventsSupervisor, BaseUILogEvents)

    def test_has_standard_agent_events(self):
        """Test that standard agent events are present."""
        assert hasattr(UILogEventsSupervisor, "ON_AGENT_FINAL_ANSWER")
        assert hasattr(UILogEventsSupervisor, "ON_TOOL_EXECUTION_SUCCESS")
        assert hasattr(UILogEventsSupervisor, "ON_TOOL_EXECUTION_FAILED")

    def test_all_events_are_unique(self):
        """Test that all event values are unique."""
        values = [e.value for e in UILogEventsSupervisor]
        assert len(values) == len(set(values))

    def test_event_count(self):
        """Test the total number of events."""
        assert len(UILogEventsSupervisor) == 8
