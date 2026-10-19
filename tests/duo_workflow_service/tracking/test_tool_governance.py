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

    @pytest.mark.parametrize(
        ("event_name", "outcome", "expected_outcome"),
        [
            (EventEnum.WORKFLOW_TOOL_BLOCKED, None, "none"),
            (EventEnum.WORKFLOW_TOOL_APPROVAL_REQUESTED, None, "none"),
            (EventEnum.WORKFLOW_TOOL_APPROVAL_RESOLVED, "rejection", "rejection"),
        ],
        ids=["blocked", "approval requested", "approval resolved"],
    )
    def test_counts_the_governance_decision(
        self, internal_event_client, event_name, outcome, expected_outcome
    ):
        """Every governance decision is counted, not just the denied ones.

        A lost pre-approval shows up as a rise in approval requests, not as a block, so that counter is how it gets
        noticed.
        """
        with patch(
            "duo_workflow_service.tracking.tool_governance.duo_workflow_metrics"
        ) as mock_metrics:
            track_tool_governance_event(
                internal_event_client,
                event_name,
                flow_type="chat",
                flow_id="flow-1",
                tool_name="run_command",
                outcome=outcome,
            )

        mock_metrics.count_tool_governance_event.assert_called_once_with(
            flow_type="chat",
            event=event_name.value,
            outcome=expected_outcome,
        )

    def test_counter_failure_still_emits_the_event(self, internal_event_client):
        """The counter must not be able to suppress the internal event."""
        with (
            patch(
                "duo_workflow_service.tracking.tool_governance.duo_workflow_metrics"
            ) as mock_metrics,
            patch(
                "duo_workflow_service.tracking.tool_governance.log_exception"
            ) as mock_log_exception,
        ):
            mock_metrics.count_tool_governance_event.side_effect = RuntimeError("boom")

            track_tool_governance_event(
                internal_event_client,
                EventEnum.WORKFLOW_TOOL_BLOCKED,
                flow_type="chat",
                flow_id="flow-1",
                tool_name="run_command",
            )

        internal_event_client.track_event.assert_called_once()
        mock_log_exception.assert_called_once()

    def test_event_failure_still_counts_the_decision(self, internal_event_client):
        """And the event must not be able to suppress the counter."""
        internal_event_client.track_event.side_effect = RuntimeError("boom")

        with (
            patch(
                "duo_workflow_service.tracking.tool_governance.duo_workflow_metrics"
            ) as mock_metrics,
            patch("duo_workflow_service.tracking.tool_governance.log_exception"),
        ):
            track_tool_governance_event(
                internal_event_client,
                EventEnum.WORKFLOW_TOOL_BLOCKED,
                flow_type="chat",
                flow_id="flow-1",
                tool_name="run_command",
            )

        mock_metrics.count_tool_governance_event.assert_called_once()
