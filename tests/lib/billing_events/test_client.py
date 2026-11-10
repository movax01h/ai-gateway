from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock

import pytest
import requests
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from pydantic import SecretStr, ValidationError
from snowplow_tracker import SelfDescribingJson, Snowplow

from lib.billing_events import (
    BillingEvent,
    BillingEventContext,
    BillingEventDropReason,
    BillingEventsClient,
)
from lib.feature_flags import FeatureFlag, current_feature_flag_context
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)
from lib.usage_quota.client import SKIP_USAGE_CUTOFF_CLAIM

BASE_BILLING_CONTEXT_SCHEMA: Dict[str, Any] = {
    "event_id": None,
    "event_type": None,
    "unit_of_measure": None,
    "quantity": None,
    "realm": None,
    "timestamp": None,
    "instance_id": None,
    "instance_version": None,
    "unique_instance_id": "",
    "host_name": None,
    "project_id": None,
    "namespace_id": None,
    "subject": None,
    "global_user_id": None,
    "root_namespace_id": None,
    "correlation_id": None,
    "seat_ids": None,
    "metadata": {},
    "deployment_type": None,
    "organization_id": None,
    "subject_type": None,
}


@pytest.fixture(scope="class", autouse=True)
def cleanup():
    """Ensure Snowplow cache is reset between tests."""
    yield
    Snowplow.reset()


@pytest.fixture(name="mock_dependencies")
def mock_dependencies_fixture():
    with (
        mock.patch("snowplow_tracker.Tracker.track") as mock_track,
        mock.patch(
            "snowplow_tracker.events.StructuredEvent.__init__", return_value=None
        ) as mock_structured_event_init,
        mock.patch("lib.billing_events.client.uuid") as mock_uuid,
        mock.patch("lib.billing_events.client.datetime") as mock_datetime,
    ):
        mock_uuid.uuid4.return_value.__str__ = mock.Mock(
            return_value="12345678-1234-5678-9012-123456789012"
        )
        mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T10:00:00"
        yield {
            "track": mock_track,
            "structured_event_init": mock_structured_event_init,
            "uuid": mock_uuid,
            "datetime": mock_datetime,
        }


@pytest.fixture(name="client")
def client_fixture():
    """Fixture for an enabled BillingEventsClient with mocked initializers."""
    with (
        mock.patch("snowplow_tracker.Tracker.__init__", return_value=None),
        mock.patch(
            "snowplow_tracker.emitters.AsyncEmitter.__init__", return_value=None
        ),
    ):
        yield BillingEventsClient(
            enabled=True,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )


@pytest.fixture(name="user")
def user_fixture():
    return CloudConnectorUser(
        authenticated=True, claims=UserClaims(gitlab_instance_uid="abc")
    )


class TestBillingEventsClient:
    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_initialization(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        mock_emitter_init.assert_called_once()
        mock_tracker_init.assert_called_once()

        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["batch_size"] == 3
        assert emitter_args["thread_count"] == 2
        assert emitter_args["endpoint"] == "https://billing.local"

        tracker_args = mock_tracker_init.call_args[1]
        assert tracker_args["app_id"] == "gitlab_ai_gateway-billing"
        assert tracker_args["namespace"] == "gl"
        assert len(tracker_args["emitters"]) == 1
        assert tracker_args["emitters"][0] is client._emitter

    @pytest.mark.parametrize(
        "api_key,expected_header",
        [
            pytest.param(None, None, id="no_api_key"),
            pytest.param(SecretStr("glsa-key"), "Bearer glsa-key", id="api_key"),
        ],
    )
    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_initialization_session_bearer_token(
        self, mock_emitter_init, mock_tracker_init, api_key, expected_header
    ):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=1,
            thread_count=1,
            internal_event_client=MagicMock(spec=InternalEventsClient),
            api_key=api_key,
        )

        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["session"] is client._session
        assert client._session.headers.get("Authorization") == expected_header

    @pytest.mark.parametrize(
        "event, unit_of_measure, quantity, metadata, category, kwargs",
        [
            (
                BillingEvent.AIGW_PROXY_USE,
                "tokens",
                100.0,
                None,
                None,
                {},
            ),
            (
                BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
                "requests",
                5.0,
                {"model": "claude-3", "feature": "completion"},
                "code_suggestions_category",
                {"project_id": 123, "namespace_id": 456},
            ),
            (
                BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS,
                "messages",
                1.0,
                {"session_id": "session-123"},
                "chat_category",
                {"global_user_id": "user-789"},
            ),
        ],
    )
    def test_track_billing_event(
        self,
        client,
        mock_dependencies,
        event,
        unit_of_measure,
        quantity,
        metadata,
        category,
        kwargs,
    ):
        user = CloudConnectorUser(
            authenticated=True,
            claims=UserClaims(gitlab_instance_uid="test-instance-uid"),
        )

        event_context = EventContext(
            realm="user",
            instance_id="instance-123",
            host_name="gitlab.example.com",
            project_id=kwargs.get("project_id"),
            namespace_id=kwargs.get("namespace_id"),
            root_namespace_id=kwargs.get("root_namespace_id"),
            global_user_id=kwargs.get("global_user_id"),
            user_id=kwargs.get("user_id"),
            correlation_id="corr-123",
            deployment_type=".com",
            user_type="human",
            feature_enablement_type="duo_pro",
            organization_id=kwargs.get("organization_id"),
        )
        current_event_context.set(event_context)

        expected_context_data = {
            **BASE_BILLING_CONTEXT_SCHEMA,
            "event_id": "12345678-1234-5678-9012-123456789012",
            "event_type": event.value,
            "unit_of_measure": unit_of_measure,
            "quantity": quantity,
            "realm": "user",
            "timestamp": "2023-12-01T10:00:00",
            "instance_id": "instance-123",
            "host_name": "gitlab.example.com",
            "project_id": kwargs.get("project_id"),
            "namespace_id": kwargs.get("namespace_id"),
            "root_namespace_id": kwargs.get("root_namespace_id"),
            "subject": kwargs.get("user_id"),
            "global_user_id": kwargs.get("global_user_id"),
            "correlation_id": "corr-123",
            "metadata": {**(metadata or {}), "is_gitlab_team_member": False},
            "unique_instance_id": "test-instance-uid",
            "deployment_type": ".com",
            "subject_type": "human",
            "assignments": ["duo_pro"],
            "organization_id": kwargs.get("organization_id"),
        }

        tracked = client.track_billing_event(
            user=user,
            event=event,
            unit_of_measure=unit_of_measure,
            quantity=quantity,
            metadata=metadata,
            category=category,
        )

        assert tracked is True
        mock_dependencies["track"].assert_called_once()
        mock_dependencies["structured_event_init"].assert_called_once()

        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        assert event_init_args["action"] == event.value

        context = event_init_args["context"][0]
        assert isinstance(context, SelfDescribingJson)
        assert context.schema == client.BILLING_CONTEXT_SCHEMA
        assert context.data == expected_context_data

    def test_track_billing_event_disabled_client(self, user):
        client = BillingEventsClient(
            enabled=False,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        assert not hasattr(client, "snowplow_tracker")

        try:
            tracked = client.track_billing_event(
                user=user,
                event=BillingEvent.AIGW_PROXY_USE,
                category=__name__,
                unit_of_measure="tokens",
                quantity=100.0,
            )
        except Exception as e:
            pytest.fail(f"Disabled client raised an unexpected exception: {e}")

        assert tracked is False

    def test_track_billing_event_negative_quantity(self, client, user):
        with mock.patch.object(client.snowplow_tracker, "track") as mock_track:
            tracked = client.track_billing_event(
                user=user,
                event=BillingEvent.AIGW_PROXY_USE,
                category=__name__,
                unit_of_measure="tokens",
                quantity=-100.0,
            )
            mock_track.assert_not_called()
            assert tracked is False

    def test_track_billing_event_with_empty_metadata(
        self, client, user, mock_dependencies
    ):
        current_event_context.set(EventContext())

        client.track_billing_event(
            user=user,
            event=BillingEvent.AIGW_PROXY_USE,
            category=__name__,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata=None,
        )

        mock_dependencies["track"].assert_called_once()
        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]

        assert context.data["metadata"] == {"is_gitlab_team_member": False}

        # None metadata must also be forwarded to the internal event client
        client.internal_event_client.track_event.assert_called_once_with(
            event_name="usage_billing_event",
            category=__name__,
            additional_properties=InternalEventAdditionalProperties(
                property=BillingEvent.AIGW_PROXY_USE.value,
                label="12345678-1234-5678-9012-123456789012",
                metadata={"is_gitlab_team_member": False},
            ),
        )

    @pytest.mark.parametrize(
        "claims,expected_flag",
        [
            (UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}), True),
            (UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: False}), False),
            (UserClaims(extra={"other_claim": True}), False),
            (UserClaims(extra=None), False),
            (None, False),
        ],
    )
    def test_track_billing_event_sets_is_gitlab_team_member(
        self, client, mock_dependencies, claims, expected_flag
    ):
        """The skip_usage_cutoff claim is surfaced as metadata.is_gitlab_team_member."""
        current_event_context.set(EventContext())

        client.track_billing_event(
            user=CloudConnectorUser(authenticated=True, claims=claims),
            event=BillingEvent.AIGW_PROXY_USE,
            category=__name__,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata={"model": "claude-3"},
        )

        expected_metadata = {
            "model": "claude-3",
            "is_gitlab_team_member": expected_flag,
        }

        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        assert context.data["metadata"] == expected_metadata

        # The same enriched metadata is forwarded to the internal event client
        client.internal_event_client.track_event.assert_called_once_with(
            event_name="usage_billing_event",
            category=__name__,
            additional_properties=InternalEventAdditionalProperties(
                property=BillingEvent.AIGW_PROXY_USE.value,
                label="12345678-1234-5678-9012-123456789012",
                metadata=expected_metadata,
            ),
        )

    def test_track_billing_event_team_member_false_with_internal_ai_usage_flag(
        self, client, mock_dependencies
    ):
        """The internal_ai_usage flag turns the skip claim off in billing metadata."""
        current_event_context.set(EventContext())
        current_feature_flag_context.set({FeatureFlag.INTERNAL_AI_USAGE.value})

        client.track_billing_event(
            user=CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
            ),
            event=BillingEvent.AIGW_PROXY_USE,
            category=__name__,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata={"model": "claude-3"},
        )

        expected_metadata = {"model": "claude-3", "is_gitlab_team_member": False}
        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        assert context.data["metadata"] == expected_metadata

        client.internal_event_client.track_event.assert_called_once_with(
            event_name="usage_billing_event",
            category=__name__,
            additional_properties=InternalEventAdditionalProperties(
                property=BillingEvent.AIGW_PROXY_USE.value,
                label="12345678-1234-5678-9012-123456789012",
                metadata=expected_metadata,
            ),
        )

    def test_track_billing_event_overrides_caller_supplied_team_member_flag(
        self, client, mock_dependencies
    ):
        """A caller-provided is_gitlab_team_member value never wins over the claim."""
        current_event_context.set(EventContext())

        client.track_billing_event(
            user=CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
            ),
            event=BillingEvent.AIGW_PROXY_USE,
            category=__name__,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata={"is_gitlab_team_member": False},
        )

        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        assert context.data["metadata"] == {"is_gitlab_team_member": True}

    @pytest.mark.usefixtures("mock_dependencies")
    def test_track_billing_event_does_not_mutate_caller_metadata(self, client, user):
        """Enrichment must not leak back into the dict owned by the caller."""
        current_event_context.set(EventContext())
        metadata = {"model": "claude-3"}

        client.track_billing_event(
            user=user,
            event=BillingEvent.AIGW_PROXY_USE,
            category=__name__,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata=metadata,
        )

        assert metadata == {"model": "claude-3"}

    def test_billing_event_context_creation_with_internal_context(
        self, client, user, mock_dependencies
    ):
        internal_context = EventContext(
            environment="production",
            realm="project",
            instance_id="gitlab-instance-456",
            host_name="gitlab.company.com",
            project_id=789,
            namespace_id=101,
            root_namespace_id=450,
            global_user_id="global-user-456",
            user_id="user-456",
            correlation_id="request-789",
            deployment_type=".com",
        )
        current_event_context.set(internal_context)

        client.track_billing_event(
            user=user,
            event=BillingEvent.DAP_FLOW_ON_COMPLETION,
            category=__name__,
            unit_of_measure="executions",
            quantity=1.0,
            metadata={"workflow_type": "code_review"},
        )

        mock_dependencies["track"].assert_called_once()
        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        billing_data = context.data

        assert billing_data["event_type"] == BillingEvent.DAP_FLOW_ON_COMPLETION.value
        assert billing_data["realm"] == "project"
        assert billing_data["instance_id"] == "gitlab-instance-456"
        assert billing_data["subject"] == "user-456"
        assert billing_data["global_user_id"] == "global-user-456"
        assert billing_data["correlation_id"] == "request-789"
        assert billing_data["metadata"] == {
            "workflow_type": "code_review",
            "is_gitlab_team_member": False,
        }
        assert billing_data["timestamp"] == "2023-12-01T10:00:00"
        assert billing_data["event_id"] == "12345678-1234-5678-9012-123456789012"
        assert billing_data["deployment_type"] == ".com"

    def test_track_billing_event_tracker_exception(self, client):
        """Test that exceptions from snowplow_tracker.track are handled gracefully."""
        user = CloudConnectorUser(
            authenticated=True, claims=UserClaims(gitlab_instance_uid="abc")
        )
        current_event_context.set(EventContext())

        with mock.patch.object(client.snowplow_tracker, "track") as mock_track:
            mock_track.side_effect = Exception("Network error")

            try:
                tracked = client.track_billing_event(
                    user=user,
                    event=BillingEvent.AIGW_PROXY_USE,
                    category=__name__,
                    unit_of_measure="tokens",
                    quantity=100.0,
                )
            except Exception as e:
                pytest.fail(f"Failed to send billing event: {e}")

            assert tracked is False

    def test_track_billing_event_internal_event_exception(
        self, client, user, mock_dependencies
    ):
        """Test that a failure in the internal analytics event still reports the billing event as tracked, since the
        Snowplow billing event was already sent."""
        current_event_context.set(EventContext())
        client.internal_event_client.track_event.side_effect = Exception(
            "Internal events down"
        )

        try:
            tracked = client.track_billing_event(
                user=user,
                event=BillingEvent.AIGW_PROXY_USE,
                category=__name__,
                unit_of_measure="tokens",
                quantity=100.0,
            )
        except Exception as e:
            pytest.fail(f"Internal event failure leaked out: {e}")

        assert tracked is True
        mock_dependencies["track"].assert_called_once()
        client.internal_event_client.track_event.assert_called_once()

    @pytest.mark.usefixtures("mock_dependencies")
    def test_internal_events_client_track_event_called_with_correct_parameters(
        self, client, user
    ):
        """Test that internal_events_client.track_event is called with correct parameters."""
        event = BillingEvent.AIGW_PROXY_USE
        category = "test_category"
        metadata = {"workflow_type": "code_review"}

        client.track_billing_event(
            user=user,
            event=event,
            category=category,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata=metadata,
        )

        client.internal_event_client.track_event.assert_called_once_with(
            event_name="usage_billing_event",
            category=category,
            additional_properties=InternalEventAdditionalProperties(
                property=event.value,
                label="12345678-1234-5678-9012-123456789012",
                metadata={**metadata, "is_gitlab_team_member": False},
            ),
        )

    def test_internal_events_client_not_called_when_billing_disabled(self, user):
        """Test that internal_events_client.track_event is not called when billing is disabled."""
        with (
            mock.patch("snowplow_tracker.Tracker.__init__", return_value=None),
            mock.patch(
                "snowplow_tracker.emitters.AsyncEmitter.__init__", return_value=None
            ),
        ):
            client = BillingEventsClient(
                enabled=False,
                endpoint="https://billing.local",
                app_id="gitlab_ai_gateway",
                namespace="gl",
                batch_size=3,
                thread_count=2,
                internal_event_client=MagicMock(spec=InternalEventsClient),
            )

            client.track_billing_event(
                user=user,
                event=BillingEvent.AIGW_PROXY_USE,
                category="test_category",
                unit_of_measure="tokens",
                quantity=100.0,
            )

            client.internal_event_client.track_event.assert_not_called()

    def test_internal_events_client_not_called_on_tracker_exception(
        self, client, user, mock_dependencies
    ):
        """Test that internal_events_client.track_event is not called when tracker raises exception."""
        mock_dependencies["track"].side_effect = Exception("Network error")

        client.track_billing_event(
            user=user,
            event=BillingEvent.AIGW_PROXY_USE,
            category="test_category",
            unit_of_measure="tokens",
            quantity=100.0,
        )

        client.internal_event_client.track_event.assert_not_called()

    def test_track_billing_event_with_instance_version(
        self, client, user, mock_dependencies
    ):
        """Test that instance_version is properly included in billing context when provided."""
        instance_version = "16.8.0-pre"
        internal_context = EventContext(
            realm="user",
            instance_id="instance-123",
            instance_version=instance_version,
            host_name="gitlab.example.com",
            correlation_id="corr-456",
            deployment_type="self-managed",
        )
        current_event_context.set(internal_context)

        client.track_billing_event(
            user=user,
            event=BillingEvent.AIGW_PROXY_USE,
            category="test_category",
            unit_of_measure="tokens",
            quantity=50.0,
        )

        mock_dependencies["track"].assert_called_once()
        mock_dependencies["structured_event_init"].assert_called_once()

        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        billing_data = context.data

        assert billing_data["instance_version"] == instance_version
        assert billing_data["instance_id"] == "instance-123"
        assert billing_data["deployment_type"] == "self-managed"

    @pytest.mark.parametrize(
        "subject_type",
        ["human", "service_account", "bot", None],
    )
    def test_track_billing_event_with_subject_type(
        self, client, user, mock_dependencies, subject_type
    ):
        """Test that EventContext.user_type is mapped to subject_type on the billing context."""
        internal_context = EventContext(
            realm="user",
            instance_id="instance-123",
            host_name="gitlab.example.com",
            correlation_id="corr-456",
            deployment_type="self-managed",
            user_type=subject_type,
        )
        current_event_context.set(internal_context)

        client.track_billing_event(
            user=user,
            event=BillingEvent.AIGW_PROXY_USE,
            category="test_category",
            unit_of_measure="tokens",
            quantity=50.0,
        )

        mock_dependencies["track"].assert_called_once()
        mock_dependencies["structured_event_init"].assert_called_once()

        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        billing_data = context.data

        # Schema 1-0-3 declares subject_type as nullable; the key must be
        # present in the wire payload even when None so the iglu validator
        # sees `null` rather than a missing field.
        assert "subject_type" in billing_data
        assert billing_data["subject_type"] == subject_type
        assert context.schema == "iglu:com.gitlab/billable_usage/jsonschema/1-0-3"

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_initialization_with_callbacks(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        mock_emitter_init.assert_called_once()
        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["on_success"].__func__ is BillingEventsClient._on_success
        assert emitter_args["on_success"].__self__ is client
        assert emitter_args["on_failure"].__func__ is BillingEventsClient._on_failure
        assert emitter_args["on_failure"].__self__ is client

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_on_failure_logs_warning_and_failed_events(
        self, mock_emitter_init, mock_tracker_init
    ):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        failed_events = [
            {"se_ac": "test_event_1", "eid": "event-id-1"},
            {"se_ac": "test_event_2", "eid": "event-id-2"},
        ]

        with (
            mock.patch.object(client._logger, "warning") as mock_warning,
            mock.patch.object(client, "_log_event") as mock_log_event,
            mock.patch.object(
                client._emitter, "last_failure_is_retryable", return_value=True
            ),
        ):
            client._on_failure(succeeded_count=5, failed_events=failed_events)

            mock_warning.assert_called_once_with(
                "Failed to track billing events",
                succeeded_count=5,
                failed_count=2,
                will_retry=True,
            )
            assert mock_log_event.call_count == 2
            mock_log_event.assert_any_call(failed_events[0], success=False)
            mock_log_event.assert_any_call(failed_events[1], success=False)

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_on_success_logs_info_and_sent_events(
        self, mock_emitter_init, mock_tracker_init
    ):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        sent_events = [
            {"se_ac": "test_event_1", "eid": "event-id-1"},
            {"se_ac": "test_event_2", "eid": "event-id-2"},
        ]

        with mock.patch.object(client._logger, "info") as mock_info:
            with mock.patch.object(client, "_log_event") as mock_log_event:
                client._on_success(sent_events=sent_events)

                mock_info.assert_called_once_with(
                    "Successfully sent billing events",
                    sent_count=2,
                )
                assert mock_log_event.call_count == 2
                mock_log_event.assert_any_call(sent_events[0], success=True)
                mock_log_event.assert_any_call(sent_events[1], success=True)

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_log_event_logs_failure_details(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        event_payload = {
            "se_ac": "test_event",
            "se_la": "test_label",
            "se_pr": "test_property",
            "eid": "event-id-1",
        }

        with mock.patch.object(client._logger, "error") as mock_error:
            client._log_event(event_payload, success=False)

            mock_error.assert_called_once_with(
                "Billing event failed to send",
                event_name="test_event",
                event_id="event-id-1",
                label="test_label",
                property="test_property",
            )

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_log_event_logs_success_details(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        event_payload = {
            "se_ac": "test_event",
            "se_la": "test_label",
            "se_pr": "test_property",
            "eid": "event-id-1",
        }

        with mock.patch.object(client._logger, "info") as mock_info:
            client._log_event(event_payload, success=True)

            mock_info.assert_called_once_with(
                "Billing event sent successfully",
                event_name="test_event",
                event_id="event-id-1",
                label="test_label",
                property="test_property",
            )

    @pytest.mark.parametrize(
        "queue_size,buffer_size,expected_queue,expected_buffer,expected_total",
        [
            (15, 3, 15, 3, 18),
            (None, 5, -1, 5, -1),
            (10, None, 10, -1, -1),
            (None, None, -1, -1, -1),
        ],
    )
    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_shutdown_logs_stats(
        self,
        mock_emitter_init: MagicMock,
        mock_tracker_init: MagicMock,
        queue_size: int | None,
        buffer_size: int | None,
        expected_queue: int,
        expected_buffer: int,
        expected_total: int,
    ):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = BillingEventsClient(
            enabled=True,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=1,
            thread_count=1,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        if queue_size is not None:
            mock_queue = MagicMock()
            mock_queue.qsize.return_value = queue_size
            client._emitter.queue = mock_queue

        if buffer_size is not None:
            mock_event_store = MagicMock()
            mock_event_store.size.return_value = buffer_size
            client._emitter.event_store = mock_event_store

        with (
            mock.patch.object(client.snowplow_tracker, "flush") as mock_flush,
            mock.patch.object(client._session, "close") as mock_close,
            mock.patch.object(client._logger, "info") as mock_info,
        ):
            client.shutdown()
            mock_flush.assert_called_once()
            mock_close.assert_called_once()
            mock_info.assert_called_once_with(
                "Shutting down billing events client",
                emitter_queue_size=expected_queue,
                emitter_buffer_size=expected_buffer,
                emitter_total_pending=expected_total,
            )


@pytest.fixture(name="live_client")
def live_client_fixture(metrics):
    """Enabled client with a real Snowplow emitter and event store; only the HTTP request and retry timer are mocked."""
    client = BillingEventsClient(
        enabled=True,
        endpoint="https://billing.local",
        app_id="gitlab_ai_gateway-billing",
        namespace="gl",
        batch_size=1,
        thread_count=1,
        internal_event_client=MagicMock(spec=InternalEventsClient),
        metrics=metrics,
    )
    with mock.patch.object(client._emitter, "_set_retry_timer"):
        yield client
    client._session.close()


@pytest.fixture(name="mock_post")
def mock_post_fixture(live_client):
    with mock.patch.object(live_client._emitter.request_method, "post") as mock_post:
        yield mock_post


def _sample(registry, name, **labels):
    return registry.get_sample_value(name, labels) or 0


def _emitted(registry, event_type):
    return _sample(registry, "billing_events_emitted_total", event_type=event_type)


def _delivered(registry, event_type):
    return _sample(registry, "billing_events_delivered_total", event_type=event_type)


def _dropped(registry, reason, event_type):
    return _sample(
        registry,
        "billing_events_dropped_total",
        reason=reason.value,
        event_type=event_type,
    )


def _no_drops(registry, event_type):
    return all(
        _dropped(registry, reason, event_type) == 0 for reason in BillingEventDropReason
    )


def _payload(event, eid="event-id-1"):
    return {"se_ac": event.value, "eid": eid}


class TestBillingEventsClientMetrics:
    @pytest.mark.usefixtures("mock_dependencies")
    @pytest.mark.parametrize(
        ("method", "expected_emitted"),
        [
            # Direct callers: the client owns the emitted count.
            ("track_billing_event", 1),
            # BillingEventService has already counted the attempt, so it must not be counted again.
            ("send_billing_event", 0),
        ],
    )
    def test_emitted_is_counted_only_by_track_billing_event(
        self, metered_client, user, registry, method, expected_emitted
    ):
        current_event_context.set(EventContext())

        assert getattr(metered_client, method)(
            user=user,
            event=BillingEvent.CODE_SUGGESTIONS_CODE_COMPLETIONS,
            category=__name__,
        )

        assert _emitted(registry, "code_completions") == expected_emitted
        assert _no_drops(registry, "code_completions")

    def test_disabled_client_counts_nothing(self, metrics, registry, user):
        client = BillingEventsClient(
            enabled=False,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=1,
            thread_count=1,
            internal_event_client=MagicMock(spec=InternalEventsClient),
            metrics=metrics,
        )

        client.track_billing_event(
            user=user, event=BillingEvent.AIGW_PROXY_USE, category=__name__
        )

        assert _emitted(registry, "ai_gateway_proxy_use") == 0

    @pytest.mark.usefixtures("mock_dependencies")
    def test_invalid_quantity_counts_missing_field(
        self, metered_client, user, registry
    ):
        current_event_context.set(EventContext())

        tracked = metered_client.track_billing_event(
            user=user,
            event=BillingEvent.AIGW_PROXY_USE,
            category=__name__,
            quantity=0,
        )

        assert tracked is False
        assert _emitted(registry, "ai_gateway_proxy_use") == 1
        assert (
            _dropped(
                registry, BillingEventDropReason.MISSING_FIELD, "ai_gateway_proxy_use"
            )
            == 1
        )

    @pytest.mark.usefixtures("mock_dependencies")
    def test_context_validation_error_counts_missing_field_and_reraises(
        self, metered_client, user, registry
    ):
        current_event_context.set(EventContext())

        with (
            mock.patch(
                "lib.billing_events.client.BillingEventContext",
                side_effect=lambda **_: BillingEventContext.model_validate({}),
            ),
            pytest.raises(ValidationError),
        ):
            metered_client.track_billing_event(
                user=user,
                event=BillingEvent.DAP_FLOW_ON_COMPLETION,
                category=__name__,
            )

        assert (
            _dropped(
                registry,
                BillingEventDropReason.MISSING_FIELD,
                "duo_agent_platform_workflow_completion",
            )
            == 1
        )

    def test_successful_send_counts_delivered(self, live_client, mock_post, registry):
        mock_post.return_value = MagicMock(status_code=200)

        live_client._emitter.send_events(
            [
                _payload(BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS, "event-id-1"),
                _payload(BillingEvent.CODE_SUGGESTIONS_CODE_GENERATIONS, "event-id-2"),
            ]
        )

        assert _delivered(registry, "code_generations") == 2
        assert _no_drops(registry, "code_generations")

    @pytest.mark.parametrize("status_code", [400, 401, 403, 410, 422])
    def test_non_retryable_status_counts_http_error(
        self, live_client, mock_post, registry, status_code
    ):
        mock_post.return_value = MagicMock(status_code=status_code)

        live_client._emitter.send_events([_payload(BillingEvent.AIGW_PROXY_USE)])

        assert (
            _dropped(
                registry, BillingEventDropReason.HTTP_ERROR, "ai_gateway_proxy_use"
            )
            == 1
        )
        assert live_client._emitter.event_store.size() == 0
        assert _delivered(registry, "ai_gateway_proxy_use") == 0

    @pytest.mark.parametrize(
        "post_kwargs",
        [
            pytest.param(
                {"return_value": MagicMock(status_code=500)}, id="server_error"
            ),
            pytest.param(
                {"return_value": MagicMock(status_code=503)}, id="unavailable"
            ),
            pytest.param(
                {"side_effect": requests.ConnectionError("connection refused")},
                id="request_exception",
            ),
        ],
    )
    def test_retryable_failure_is_not_counted_as_dropped(
        self, live_client, mock_post, registry, post_kwargs
    ):
        """Snowplow calls on_failure on every failed attempt; a retried event is not lost, so it must not alert."""
        mock_post.configure_mock(**post_kwargs)

        live_client._emitter.send_events([_payload(BillingEvent.AIGW_PROXY_USE)])

        assert live_client._emitter.event_store.size() == 1
        assert _no_drops(registry, "ai_gateway_proxy_use")

    def test_retry_after_transient_failure_counts_delivered_once(
        self, live_client, mock_post, registry
    ):
        mock_post.side_effect = [
            MagicMock(status_code=503),
            MagicMock(status_code=200),
        ]

        live_client._emitter.send_events([_payload(BillingEvent.AIGW_PROXY_USE)])
        live_client._emitter.send_events(
            live_client._emitter.event_store.get_events_batch()
        )

        assert _delivered(registry, "ai_gateway_proxy_use") == 1
        assert _no_drops(registry, "ai_gateway_proxy_use")

    def test_unexpected_event_type_is_labelled_unknown(
        self, live_client, mock_post, registry
    ):
        mock_post.return_value = MagicMock(status_code=400)

        live_client._emitter.send_events([{"se_ac": "not-a-billing-event"}])

        assert _dropped(registry, BillingEventDropReason.HTTP_ERROR, "unknown") == 1
