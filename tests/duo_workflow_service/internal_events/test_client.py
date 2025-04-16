from unittest import mock

import pytest
from snowplow_tracker.self_describing_json import SelfDescribingJson
from snowplow_tracker.snowplow import Snowplow

from duo_workflow_service.internal_events.client import (
    InternalEventsClient,
    InternalEventsConfig,
)
from duo_workflow_service.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)


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

        config = InternalEventsConfig(
            enabled=True,
            endpoint="https://example.com",
            app_id="gitlab_duo_workflow",
            namespace="gl",
            batch_size=3,
            thread_count=2,
        )

        client = InternalEventsClient(config)

        mock_emitter_init.assert_called_once()
        mock_tracker_init.assert_called_once()

        emitter_args = mock_emitter_init.call_args[1]
        assert emitter_args["batch_size"] == 3
        assert emitter_args["thread_count"] == 2
        assert emitter_args["endpoint"] == "https://example.com"

        tracker_args = mock_tracker_init.call_args[1]
        assert tracker_args["app_id"] == "gitlab_duo_workflow"
        assert tracker_args["namespace"] == "gl"
        assert len(tracker_args["emitters"]) == 1

    @pytest.mark.parametrize(
        "event_name, additional_properties, category, expected_extra",
        [
            (
                "test_event_1",
                InternalEventAdditionalProperties(extra={"key1": "value1"}),
                "category_1",
                {"key1": "value1"},
            ),
            (
                "test_event_2",
                InternalEventAdditionalProperties(extra={"key2": "value2"}),
                "category_2",
                {"key2": "value2"},
            ),
        ],
    )
    def test_track_event(
        self,
        event_name,
        additional_properties,
        category,
        expected_extra,
    ):
        with mock.patch("snowplow_tracker.Tracker.track") as mock_track, mock.patch(
            "snowplow_tracker.events.StructuredEvent.__init__"
        ) as mock_structured_event_init:
            mock_structured_event_init.return_value = None

            # Set up current event context
            current_event_context.set(EventContext())

            config = InternalEventsConfig(
                enabled=True,
                endpoint="https://example.com",
                app_id="gitlab_duo_workflow",
                namespace="gl",
                batch_size=3,
                thread_count=2,
            )

            client = InternalEventsClient(config)

            client.track_event(
                event_name=event_name,
                additional_properties=additional_properties,
                category=category,
            )

            mock_track.assert_called_once()
            mock_structured_event_init.assert_called_once()
            event_init_args = mock_structured_event_init.call_args[1]

            assert event_init_args["category"] == category
            assert event_init_args["action"] == event_name
            assert event_init_args["label"] == additional_properties.label
            assert event_init_args["value"] == additional_properties.value
            assert event_init_args["property_"] == additional_properties.property

            context = event_init_args["context"][0]
            assert isinstance(context, SelfDescribingJson)
            assert context.to_json()["schema"] == client.STANDARD_CONTEXT_SCHEMA
