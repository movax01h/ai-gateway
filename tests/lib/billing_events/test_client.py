from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from snowplow_tracker import SelfDescribingJson, Snowplow

from lib.billing_events.client import BillingEventsClient
from lib.feature_flags import FeatureFlag, current_feature_flag_context
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)

BASE_BILLING_CONTEXT_SCHEMA: Dict[str, Any] = {
    "event_id": None,
    "event_type": None,
    "unit_of_measure": None,
    "quantity": None,
    "realm": None,
    "timestamp": None,
    "instance_id": None,
    "unique_instance_id": "",
    "host_name": None,
    "project_id": None,
    "namespace_id": None,
    "subject": None,
    "global_user_id": None,
    "root_namespace_id": None,
    "correlation_id": None,
    "seat_ids": ["TODO"],
    "metadata": {},
    "deployment_type": None,
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

        BillingEventsClient(
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

    @pytest.mark.parametrize(
        "event_type, unit_of_measure, quantity, metadata, category, kwargs",
        [
            (
                "ai_completion",
                "tokens",
                100.0,
                None,
                None,
                {},
            ),
            (
                "code_suggestions",
                "requests",
                5.0,
                {"model": "claude-3", "feature": "completion"},
                "code_suggestions_category",
                {"project_id": 123, "namespace_id": 456},
            ),
            (
                "duo_chat",
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
        event_type,
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

        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
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
            feature_enablement_type="duo_pro",
        )
        current_event_context.set(event_context)

        expected_context_data = {
            **BASE_BILLING_CONTEXT_SCHEMA,
            "event_id": "12345678-1234-5678-9012-123456789012",
            "event_type": event_type,
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
            "metadata": metadata or {},
            "unique_instance_id": "test-instance-uid",
            "deployment_type": ".com",
            "assignments": ["duo_pro"],
        }

        client.track_billing_event(
            user=user,
            event_type=event_type,
            unit_of_measure=unit_of_measure,
            quantity=quantity,
            metadata=metadata,
            category=category,
        )

        mock_dependencies["track"].assert_called_once()
        mock_dependencies["structured_event_init"].assert_called_once()

        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        assert event_init_args["action"] == event_type

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
            client.track_billing_event(
                user=user,
                event_type="ai_completion",
                category=__name__,
                unit_of_measure="tokens",
                quantity=100.0,
            )
        except Exception as e:
            pytest.fail(f"Disabled client raised an unexpected exception: {e}")

    def test_track_billing_event_negative_quantity(self, client, user):
        with mock.patch.object(client.snowplow_tracker, "track") as mock_track:
            client.track_billing_event(
                user=user,
                event_type="ai_completion",
                category=__name__,
                unit_of_measure="tokens",
                quantity=-100.0,
            )
            mock_track.assert_not_called()

    def test_track_billing_event_feature_flag_disabled(self, client, user):
        """Test that billing events are not tracked when feature flag is disabled."""
        current_feature_flag_context.set(set())
        with mock.patch.object(client.snowplow_tracker, "track") as mock_track:
            client.track_billing_event(
                user=user,
                event_type="ai_completion",
                category=__name__,
                unit_of_measure="tokens",
                quantity=100.0,
            )
            mock_track.assert_not_called()

    def test_track_billing_event_with_empty_metadata(
        self, client, user, mock_dependencies
    ):
        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
        current_event_context.set(EventContext())

        client.track_billing_event(
            user=user,
            event_type="ai_completion",
            category=__name__,
            unit_of_measure="tokens",
            quantity=100.0,
            metadata=None,
        )

        mock_dependencies["track"].assert_called_once()
        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]

        assert context.data["metadata"] == {}

    def test_billing_event_context_creation_with_internal_context(
        self, client, user, mock_dependencies
    ):
        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
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
            event_type="duo_workflow",
            category=__name__,
            unit_of_measure="executions",
            quantity=1.0,
            metadata={"workflow_type": "code_review"},
        )

        mock_dependencies["track"].assert_called_once()
        event_init_args = mock_dependencies["structured_event_init"].call_args[1]
        context = event_init_args["context"][0]
        billing_data = context.data

        assert billing_data["event_type"] == "duo_workflow"
        assert billing_data["realm"] == "project"
        assert billing_data["instance_id"] == "gitlab-instance-456"
        assert billing_data["subject"] == "user-456"
        assert billing_data["global_user_id"] == "global-user-456"
        assert billing_data["correlation_id"] == "request-789"
        assert billing_data["metadata"] == {"workflow_type": "code_review"}
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
                client.track_billing_event(
                    user=user,
                    event_type="ai_completion",
                    category=__name__,
                    unit_of_measure="tokens",
                    quantity=100.0,
                )
            except Exception as e:
                pytest.fail(f"Failed to send billing event: {e}")

    def test_internal_events_client_track_event_called_with_correct_parameters(
        self, client, user, mock_dependencies  # pylint: disable=unused-argument
    ):
        """Test that internal_events_client.track_event is called with correct parameters."""
        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
        event_type = "ai_completion"
        category = "test_category"

        client.track_billing_event(
            user=user,
            event_type=event_type,
            category=category,
            unit_of_measure="tokens",
            quantity=100.0,
        )

        client.internal_event_client.track_event.assert_called_once_with(
            event_name="usage_billing_event",
            category=category,
            additional_properties=InternalEventAdditionalProperties(
                property=event_type, label="12345678-1234-5678-9012-123456789012"
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
                event_type="ai_completion",
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
            event_type="ai_completion",
            category="test_category",
            unit_of_measure="tokens",
            quantity=100.0,
        )

        client.internal_event_client.track_event.assert_not_called()

    def test_gitlab_team_member_subject_is_hashed(
        self, client, user, mock_dependencies
    ):
        """Test that GitLab team member global_user_id is hashed to integer in subject field."""
        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
        current_event_context.set(
            EventContext(
                user_id="12345",
                global_user_id="global-user-abc-123",
                is_gitlab_team_member=True,
            )
        )

        client.track_billing_event(
            user=user,
            event_type="ai_completion",
            category="test",
            unit_of_measure="tokens",
            quantity=100.0,
        )

        event_args = mock_dependencies["structured_event_init"].call_args[1]
        subject = event_args["context"][0].data["subject"]

        # Subject should be a string representation of a hashed integer derived from global_user_id
        assert isinstance(subject, str)
        assert subject.isdigit()
        assert int(subject) > 0

    def test_non_gitlab_team_member_subject_not_hashed(
        self, client, user, mock_dependencies
    ):
        """Test that non-GitLab team member user_id is NOT hashed in subject field."""
        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
        current_event_context.set(
            EventContext(user_id="12345", is_gitlab_team_member=False)
        )

        client.track_billing_event(
            user=user,
            event_type="ai_completion",
            category="test",
            unit_of_measure="tokens",
            quantity=100.0,
        )

        event_args = mock_dependencies["structured_event_init"].call_args[1]
        subject = event_args["context"][0].data["subject"]

        assert subject == "12345"

    def test_gitlab_team_member_with_empty_global_user_id(
        self, client, user, mock_dependencies
    ):
        """Test that empty global_user_id returns -1 in subject field."""
        current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
        current_event_context.set(
            EventContext(
                user_id="12345", global_user_id=None, is_gitlab_team_member=True
            )
        )

        client.track_billing_event(
            user=user,
            event_type="ai_completion",
            category="test",
            unit_of_measure="tokens",
            quantity=100.0,
        )

        event_args = mock_dependencies["structured_event_init"].call_args[1]
        subject = event_args["context"][0].data["subject"]

        assert subject == "-1"

    def test_use_global_user_id_for_team_members_disabled(
        self, user, mock_dependencies
    ):
        """Test that GitLab team member user_id is NOT hashed when use_global_user_id_for_team_members is False."""
        with (
            mock.patch("snowplow_tracker.Tracker.__init__", return_value=None),
            mock.patch(
                "snowplow_tracker.emitters.AsyncEmitter.__init__", return_value=None
            ),
        ):
            client = BillingEventsClient(
                enabled=True,
                endpoint="https://billing.local",
                app_id="gitlab_ai_gateway-billing",
                namespace="gl",
                batch_size=3,
                thread_count=2,
                internal_event_client=MagicMock(spec=InternalEventsClient),
                use_global_user_id_for_team_members=False,
            )

            current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
            current_event_context.set(
                EventContext(
                    user_id="12345",
                    global_user_id="global-user-abc-123",
                    is_gitlab_team_member=True,
                )
            )

            client.track_billing_event(
                user=user,
                event_type="ai_completion",
                category="test",
                unit_of_measure="tokens",
                quantity=100.0,
            )

            event_args = mock_dependencies["structured_event_init"].call_args[1]
            subject = event_args["context"][0].data["subject"]

            # Subject should be the user_id, not hashed
            assert subject == "12345"

    def test_use_global_user_id_for_team_members_enabled(self, user, mock_dependencies):
        """Test that GitLab team member global_user_id IS hashed when use_global_user_id_for_team_members is True."""
        with (
            mock.patch("snowplow_tracker.Tracker.__init__", return_value=None),
            mock.patch(
                "snowplow_tracker.emitters.AsyncEmitter.__init__", return_value=None
            ),
        ):
            client = BillingEventsClient(
                enabled=True,
                endpoint="https://billing.local",
                app_id="gitlab_ai_gateway-billing",
                namespace="gl",
                batch_size=3,
                thread_count=2,
                internal_event_client=MagicMock(spec=InternalEventsClient),
                use_global_user_id_for_team_members=True,
            )

            current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
            current_event_context.set(
                EventContext(
                    user_id="12345",
                    global_user_id="global-user-abc-123",
                    is_gitlab_team_member=True,
                )
            )

            client.track_billing_event(
                user=user,
                event_type="ai_completion",
                category="test",
                unit_of_measure="tokens",
                quantity=100.0,
            )

            event_args = mock_dependencies["structured_event_init"].call_args[1]
            subject = event_args["context"][0].data["subject"]

            # Subject should be a string representation of a hashed integer
            assert subject.isdigit()
            assert int(subject) > 0
            assert subject != "12345"

    def test_use_global_user_id_for_team_members_default_value(self):
        """Test that use_global_user_id_for_team_members defaults to True."""
        client = BillingEventsClient(
            enabled=True,
            endpoint="https://billing.local",
            app_id="gitlab_ai_gateway-billing",
            namespace="gl",
            batch_size=3,
            thread_count=2,
            internal_event_client=MagicMock(spec=InternalEventsClient),
        )

        assert client.use_global_user_id_for_team_members is True

    def test_use_global_user_id_for_team_members_does_not_affect_non_team_members(
        self, user, mock_dependencies
    ):
        """Test that use_global_user_id_for_team_members does not affect non-team members."""
        with (
            mock.patch("snowplow_tracker.Tracker.__init__", return_value=None),
            mock.patch(
                "snowplow_tracker.emitters.AsyncEmitter.__init__", return_value=None
            ),
        ):
            client = BillingEventsClient(
                enabled=True,
                endpoint="https://billing.local",
                app_id="gitlab_ai_gateway-billing",
                namespace="gl",
                batch_size=3,
                thread_count=2,
                internal_event_client=MagicMock(spec=InternalEventsClient),
                use_global_user_id_for_team_members=False,
            )

            current_feature_flag_context.set({FeatureFlag.DUO_USE_BILLING_ENDPOINT})
            current_event_context.set(
                EventContext(
                    user_id="12345",
                    global_user_id="global-user-abc-123",
                    is_gitlab_team_member=False,
                )
            )

            client.track_billing_event(
                user=user,
                event_type="ai_completion",
                category="test",
                unit_of_measure="tokens",
                quantity=100.0,
            )

            event_args = mock_dependencies["structured_event_init"].call_args[1]
            subject = event_args["context"][0].data["subject"]

            # Subject should be the user_id for non-team members regardless of the flag
            assert subject == "12345"
