from unittest import mock
from unittest.mock import patch

import pytest
from snowplow_tracker import SelfDescribingJson, Snowplow, StructuredEvent

from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
    tracked_internal_events,
)

BASE_CONTEXT_SCHEMA = {
    "client_name": None,
    "client_type": None,
    "client_version": None,
    "context_generated_at": None,
    "correlation_id": None,
    "environment": "development",
    "extra": None,
    "feature_enabled_by_namespace_ids": None,
    "feature_enablement_type": None,
    "global_user_id": None,
    "host_name": None,
    "input_tokens": None,
    "instance_id": None,
    "unique_instance_id": None,
    "instance_version": None,
    "interface": None,
    "is_gitlab_team_member": None,
    "model_engine": None,
    "model_name": None,
    "model_provider": None,
    "namespace_id": None,
    "ultimate_parent_namespace_id": None,
    "output_tokens": None,
    "plan": None,
    "project_id": None,
    "realm": None,
    "source": "ai-gateway-python",
    "total_tokens": None,
    "deployment_type": None,
}


class TestInternalEventsClient:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self):
        """Ensure Snowplow cache is reset between tests."""
        yield
        Snowplow.reset()

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_initialization(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = InternalEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        mock_emitter_init.assert_called_once()
        mock_tracker_init.assert_called_once()

        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["batch_size"] == 3
        assert emitter_args["thread_count"] == 2
        assert emitter_args["endpoint"] == "https://whitechoc.local"

        tracker_args = mock_tracker_init.call_args[1]
        assert tracker_args["app_id"] == "gitlab_ai_gateway"
        assert tracker_args["namespace"] == "gl"
        assert len(tracker_args["emitters"]) == 1

    @pytest.mark.parametrize(
        "event_name, additional_properties, category, kwargs",
        [
            (
                "test_event_1",
                None,
                None,
                {},
            ),
            (
                "test_event_2",
                InternalEventAdditionalProperties(extra={"key2": "value2"}),
                "category_2",
                {"model_name": "my_model"},
            ),
        ],
    )
    def test_track_event(
        self,
        event_name,
        additional_properties,
        category,
        kwargs,
    ):
        with (
            mock.patch("snowplow_tracker.Tracker.track") as mock_track,
            mock.patch(
                "snowplow_tracker.events.StructuredEvent.__init__"
            ) as mock_structured_event_init,
        ):
            mock_structured_event_init.return_value = None

            # Set up current event context
            current_event_context.set(EventContext())
            tracked_internal_events.set(set())

            client = InternalEventsClient(
                enabled=True,
                endpoint="https://whitechoc.local",
                app_id="gitlab_ai_gateway",
                namespace="gl",
                batch_size=3,
                thread_count=2,
            )

            client.track_event(
                event_name=event_name,
                additional_properties=additional_properties,
                category=category,
                **kwargs,
            )

            mock_track.assert_called_once()
            mock_structured_event_init.assert_called_once()
            event_init_args = mock_structured_event_init.call_args[1]

            assert event_init_args["category"] == category or "default_category"
            assert event_init_args["action"] == event_name
            if additional_properties:
                assert event_init_args["label"] == additional_properties.label
                assert event_init_args["value"] == additional_properties.value
                assert event_init_args["property_"] == additional_properties.property
            else:
                assert event_init_args["label"] is None
                assert event_init_args["value"] is None
                assert event_init_args["property_"] is None

            context = event_init_args["context"][0]
            assert isinstance(context, SelfDescribingJson)
            assert context.schema == client.STANDARD_CONTEXT_SCHEMA

            expected_context_data = {
                **BASE_CONTEXT_SCHEMA,
                "extra": additional_properties.extra if additional_properties else {},
                **kwargs,
            }

            assert context.data == expected_context_data
            assert tracked_internal_events.get() == {event_name}

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_initialization_with_callbacks(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = InternalEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        mock_emitter_init.assert_called_once()
        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["on_success"] == client._on_success
        assert emitter_args["on_failure"] == client._on_failure

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_on_failure_logs_warning_and_failed_events(
        self, mock_emitter_init, mock_tracker_init
    ):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = InternalEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        failed_events = [
            {"se_ac": "test_event_1", "eid": "event-id-1"},
            {"se_ac": "test_event_2", "eid": "event-id-2"},
        ]

        with mock.patch.object(client._logger, "warning") as mock_warning:
            with mock.patch.object(client, "_log_event") as mock_log_event:
                client._on_failure(succeeded_count=5, failed_events=failed_events)

                mock_warning.assert_called_once_with(
                    "Failed to track internal events",
                    succeeded_count=5,
                    failed_count=2,
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

        client = InternalEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        sent_events = [
            {"se_ac": "test_event_1", "eid": "event-id-1"},
            {"se_ac": "test_event_2", "eid": "event-id-2"},
        ]

        with mock.patch.object(client._logger, "info") as mock_info:
            with mock.patch.object(client, "_log_event") as mock_log_event:
                client._on_success(sent_events=sent_events)

                mock_info.assert_called_once_with(
                    "Successfully sent internal events",
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

        client = InternalEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        event_payload = {
            "se_ac": "test_event",
            "se_la": "test_label",
            "se_pr": "test_property",
        }

        with mock.patch.object(client._logger, "error") as mock_error:
            client._log_event(event_payload, success=False)

            mock_error.assert_called_once_with(
                "Internal event failed to send",
                event_name="test_event",
                label="test_label",
                property="test_property",
            )

    @mock.patch("snowplow_tracker.Tracker.__init__")
    @mock.patch("snowplow_tracker.emitters.AsyncEmitter.__init__")
    def test_log_event_logs_success_details(self, mock_emitter_init, mock_tracker_init):
        mock_emitter_init.return_value = None
        mock_tracker_init.return_value = None

        client = InternalEventsClient(
            enabled=True,
            endpoint="https://whitechoc.local",
            app_id="gitlab_ai_gateway",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        event_payload = {
            "se_ac": "test_event",
            "se_la": "test_label",
            "se_pr": "test_property",
        }

        with mock.patch.object(client._logger, "info") as mock_info:
            client._log_event(event_payload, success=True)

            mock_info.assert_called_once_with(
                "Internal event sent successfully",
                event_name="test_event",
                label="test_label",
                property="test_property",
            )
