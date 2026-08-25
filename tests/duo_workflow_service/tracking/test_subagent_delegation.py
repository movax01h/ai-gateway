"""Test suite for the sub-agent delegation event emitters."""

from unittest.mock import Mock, patch

import pytest

from duo_workflow_service.tracking.monitoring_context import (
    MonitoringContext,
    current_monitoring_context,
)
from duo_workflow_service.tracking.subagent_delegation import (
    DelegationRejectionReason,
    SubagentDelegationTracker,
    track_subagent_delegated,
    track_subagent_delegation_rejected,
    track_subagent_returned,
)
from lib.internal_events.event_enum import EventEnum


@pytest.fixture(name="internal_event_client")
def internal_event_client_fixture():
    """Fixture for mock internal event client."""
    return Mock()


class TestTrackSubagentDelegated:
    """Test suite for the delegation emitter."""

    def test_emits_event_with_expected_payload(self, internal_event_client):
        track_subagent_delegated(
            internal_event_client,
            flow_type="software_development",
            flow_id="flow-1",
            supervisor_name="developer_agent",
            subagent_name="review_agent",
            subsession_id=2,
            is_resume=True,
            delegation_count=3,
            parallel=False,
        )

        internal_event_client.track_event.assert_called_once()
        call_kwargs = internal_event_client.track_event.call_args.kwargs
        additional_properties = call_kwargs["additional_properties"]
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_SUBAGENT_DELEGATED.value
        assert call_kwargs["category"] == "software_development"
        assert additional_properties.label == "developer_agent"
        assert additional_properties.property == "review_agent"
        assert additional_properties.value == "flow-1"
        assert additional_properties.extra == {
            "subsession_id": 2,
            "is_resume": True,
            "delegation_count": 3,
            "parallel": False,
        }

    def test_swallows_and_logs_tracking_failures(self, internal_event_client):
        """Tracking is best-effort and must never break the delegation path that calls it."""
        internal_event_client.track_event.side_effect = RuntimeError("boom")

        with patch(
            "duo_workflow_service.tracking.subagent_delegation.log_exception"
        ) as mock_log_exception:
            track_subagent_delegated(
                internal_event_client,
                flow_type="chat",
                flow_id="flow-1",
                supervisor_name="developer_agent",
                subagent_name="review_agent",
                subsession_id=1,
                is_resume=False,
                delegation_count=1,
                parallel=False,
            )

        mock_log_exception.assert_called_once()


class TestTrackSubagentReturned:
    """Test suite for the return emitter."""

    @pytest.mark.parametrize("status", ["completed", "error"])
    def test_emits_event_with_expected_payload(self, internal_event_client, status):
        track_subagent_returned(
            internal_event_client,
            flow_type="chat",
            flow_id="flow-1",
            supervisor_name="developer_agent",
            subagent_name="review_agent",
            subsession_id=1,
            status=status,
            parallel=False,
        )

        internal_event_client.track_event.assert_called_once()
        call_kwargs = internal_event_client.track_event.call_args.kwargs
        additional_properties = call_kwargs["additional_properties"]
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_SUBAGENT_RETURNED.value
        assert call_kwargs["category"] == "chat"
        assert additional_properties.label == "developer_agent"
        assert additional_properties.property == "review_agent"
        assert additional_properties.value == "flow-1"
        assert additional_properties.extra == {
            "subsession_id": 1,
            "status": status,
            "parallel": False,
        }

    def test_swallows_and_logs_tracking_failures(self, internal_event_client):
        internal_event_client.track_event.side_effect = RuntimeError("boom")

        with patch(
            "duo_workflow_service.tracking.subagent_delegation.log_exception"
        ) as mock_log_exception:
            track_subagent_returned(
                internal_event_client,
                flow_type="chat",
                flow_id="flow-1",
                supervisor_name="developer_agent",
                subagent_name="review_agent",
                subsession_id=1,
                status="completed",
                parallel=False,
            )

        mock_log_exception.assert_called_once()


class TestTrackSubagentDelegationRejected:
    """Test suite for the rejection emitter."""

    def test_emits_event_with_expected_payload(self, internal_event_client):
        track_subagent_delegation_rejected(
            internal_event_client,
            flow_type="chat",
            flow_id="flow-1",
            supervisor_name="developer_agent",
            reason=DelegationRejectionReason.INVALID_ARGS,
            subagent_name=None,
            parallel=False,
        )

        internal_event_client.track_event.assert_called_once()
        call_kwargs = internal_event_client.track_event.call_args.kwargs
        additional_properties = call_kwargs["additional_properties"]
        assert (
            call_kwargs["event_name"]
            == EventEnum.WORKFLOW_SUBAGENT_DELEGATION_REJECTED.value
        )
        assert call_kwargs["category"] == "chat"
        assert additional_properties.label == "developer_agent"
        assert additional_properties.property == "invalid_args"
        assert additional_properties.value == "flow-1"
        assert additional_properties.extra == {
            "subagent_name": None,
            "parallel": False,
        }

    def test_reason_is_emitted_as_plain_string(self, internal_event_client):
        """A StrEnum member would serialise unpredictably; the wire value must be the slug."""
        track_subagent_delegation_rejected(
            internal_event_client,
            flow_type="chat",
            flow_id="flow-1",
            supervisor_name="developer_agent",
            reason=DelegationRejectionReason.LIMIT_REACHED,
            subagent_name="review_agent",
            parallel=True,
        )

        additional_properties = internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ]
        assert type(additional_properties.property) is str
        assert additional_properties.property == "limit_reached"
        assert additional_properties.extra["subagent_name"] == "review_agent"

    def test_swallows_and_logs_tracking_failures(self, internal_event_client):
        internal_event_client.track_event.side_effect = RuntimeError("boom")

        with patch(
            "duo_workflow_service.tracking.subagent_delegation.log_exception"
        ) as mock_log_exception:
            track_subagent_delegation_rejected(
                internal_event_client,
                flow_type="chat",
                flow_id="flow-1",
                supervisor_name="developer_agent",
                reason=DelegationRejectionReason.LIMIT_REACHED,
                subagent_name=None,
                parallel=False,
            )

        mock_log_exception.assert_called_once()


class TestSubagentDelegationTracker:
    """Test suite for the per-supervisor binder."""

    @pytest.fixture(name="make_tracker")
    def make_tracker_fixture(self, internal_event_client):
        def factory(parallel: bool):
            return SubagentDelegationTracker(
                flow_id="flow-1",
                flow_type=Mock(value="software_development"),
                internal_event_client=internal_event_client,
                supervisor_name="developer_agent",
                parallel=parallel,
            )

        return factory

    @pytest.mark.parametrize("parallel", [False, True])
    def test_binds_the_shared_fields_on_every_event(
        self, internal_event_client, make_tracker, parallel
    ):
        """No emit site restates these, so a node cannot mislabel which supervisor it is."""
        tracker = make_tracker(parallel)

        tracker.delegated(
            subagent_name="review_agent",
            subsession_id=1,
            is_resume=False,
            delegation_count=1,
        )
        tracker.returned(
            subagent_name="review_agent", subsession_id=1, status="completed"
        )
        tracker.rejected(reason=DelegationRejectionReason.LIMIT_REACHED)

        assert internal_event_client.track_event.call_count == 3
        for call in internal_event_client.track_event.call_args_list:
            additional_properties = call.kwargs["additional_properties"]
            assert call.kwargs["category"] == "software_development"
            assert additional_properties.label == "developer_agent"
            assert additional_properties.value == "flow-1"
            assert additional_properties.extra["parallel"] is parallel

    def test_forwards_the_per_call_fields(self, internal_event_client, make_tracker):
        make_tracker(True).delegated(
            subagent_name="review_agent",
            subsession_id=7,
            is_resume=True,
            delegation_count=3,
        )

        call_kwargs = internal_event_client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_SUBAGENT_DELEGATED.value
        assert call_kwargs["additional_properties"].property == "review_agent"
        assert call_kwargs["additional_properties"].extra == {
            "subsession_id": 7,
            "is_resume": True,
            "delegation_count": 3,
            "parallel": True,
        }

    def test_rejected_defaults_subagent_name_to_none(
        self, internal_event_client, make_tracker
    ):
        """The subagent is usually unknown at rejection, so callers should not have to say so."""
        make_tracker(False).rejected(reason=DelegationRejectionReason.INVALID_ARGS)

        additional_properties = internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ]
        assert additional_properties.property == "invalid_args"
        assert additional_properties.extra["subagent_name"] is None


class TestFlowRevisionExtras:
    """Tests that events identify which revision of a flow produced them."""

    @pytest.fixture(name="with_flow_identity")
    def with_flow_identity_fixture(self):
        context = MonitoringContext()
        context.set_flow_identity(
            flow_id="developer",
            flow_version="2.1.0-interactive",
            schema_version="v1",
        )
        token = current_monitoring_context.set(context)
        yield
        current_monitoring_context.reset(token)

    @pytest.mark.usefixtures("with_flow_identity")
    def test_every_event_carries_the_flow_revision(self, internal_event_client):
        tracker = SubagentDelegationTracker(
            flow_id="flow-1",
            flow_type=Mock(value="developer"),
            internal_event_client=internal_event_client,
            supervisor_name="developer_agent",
            parallel=False,
        )

        tracker.delegated(
            subagent_name="review_agent",
            subsession_id=1,
            is_resume=False,
            delegation_count=1,
        )
        tracker.returned(
            subagent_name="review_agent", subsession_id=1, status="completed"
        )
        tracker.rejected(reason=DelegationRejectionReason.LIMIT_REACHED)

        assert internal_event_client.track_event.call_count == 3
        for call in internal_event_client.track_event.call_args_list:
            extra = call.kwargs["additional_properties"].extra
            assert extra["flow_version"] == "2.1.0-interactive"
            assert extra["schema_version"] == "v1"

    @pytest.mark.usefixtures("with_flow_identity")
    def test_flow_id_is_not_emitted(self, internal_event_client):
        track_subagent_delegated(
            internal_event_client,
            flow_type="developer",
            flow_id="flow-1",
            supervisor_name="developer_agent",
            subagent_name="review_agent",
            subsession_id=1,
            is_resume=False,
            delegation_count=1,
            parallel=False,
        )

        extra = internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ].extra
        assert "flow_id" not in extra
        assert (
            internal_event_client.track_event.call_args.kwargs[
                "additional_properties"
            ].value
            == "flow-1"
        )

    def test_legacy_flows_contribute_nothing(self, internal_event_client):
        track_subagent_returned(
            internal_event_client,
            flow_type="software_development",
            flow_id="flow-1",
            supervisor_name="developer_agent",
            subagent_name="review_agent",
            subsession_id=1,
            status="completed",
            parallel=False,
        )

        extra = internal_event_client.track_event.call_args.kwargs[
            "additional_properties"
        ].extra
        assert extra == {
            "subsession_id": 1,
            "status": "completed",
            "parallel": False,
        }
