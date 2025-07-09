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
    "instance_version": None,
    "interface": None,
    "is_gitlab_team_member": None,
    "model_engine": None,
    "model_name": None,
    "model_provider": None,
    "namespace_id": None,
    "output_tokens": None,
    "plan": None,
    "project_id": None,
    "realm": None,
    "source": "ai-gateway-python",
    "total_tokens": None,
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
        with mock.patch("snowplow_tracker.Tracker.track") as mock_track, mock.patch(
            "snowplow_tracker.events.StructuredEvent.__init__"
        ) as mock_structured_event_init:
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
