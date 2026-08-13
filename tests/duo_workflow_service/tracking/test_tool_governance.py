"""Test suite for the shared tool-governance event emitter."""

from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.tracking.tool_governance import track_tool_governance_event
from lib.internal_events.event_enum import EventEnum


@pytest.fixture(name="internal_event_client")
def internal_event_client_fixture():
    """Fixture for mock internal event client."""
    return Mock()


class TestTrackToolGovernanceEvent:
    """Test suite for the shared emitter."""

    def test_emits_event_with_expected_payload(self, internal_event_client):
        track_tool_governance_event(
            internal_event_client,
            EventEnum.WORKFLOW_TOOL_BLOCKED,
            flow_type="chat",
            flow_id="flow-1",
            tool_name="denied_tool",
        )

        internal_event_client.track_event.assert_called_once()
        call_kwargs = internal_event_client.track_event.call_args.kwargs
        additional_properties = call_kwargs["additional_properties"]
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_TOOL_BLOCKED.value
        assert call_kwargs["category"] == "chat"
        assert additional_properties.label == "chat"
        assert additional_properties.property == "denied_tool"
        assert additional_properties.value == "flow-1"
        assert additional_properties.extra == {"tool_name": "denied_tool"}

    def test_omits_tool_name_when_unknown(self, internal_event_client):
        track_tool_governance_event(
            internal_event_client,
            EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED,
            flow_type="chat",
            flow_id="flow-1",
            outcome="approval",
        )

        additional_properties = internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ]
        assert additional_properties.property == "approval"
        assert additional_properties.extra == {}

    def test_swallows_and_logs_tracking_failures(self, internal_event_client):
        """Tracking is best-effort and must never break the enforcement paths that call it."""
        internal_event_client.track_event.side_effect = RuntimeError("boom")

        with patch(
            "duo_workflow_service.tracking.tool_governance.log_exception"
        ) as mock_log_exception:
            track_tool_governance_event(
                internal_event_client,
                EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED,
                flow_type="chat",
                flow_id="flow-1",
                tool_name="run_command",
                outcome="approval",
            )

        internal_event_client.track_event.assert_called_once()
        mock_log_exception.assert_called_once()
