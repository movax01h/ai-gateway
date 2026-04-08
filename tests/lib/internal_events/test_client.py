"""Tests for InternalEventsClient AIContext extraction."""

from unittest.mock import Mock, patch

import pytest

from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)


def _find_ai_context(structured_event):
    """Extract ai_context data from a structured event's context list.

    Args:
        structured_event: The Snowplow structured event object.

    Returns:
        The ai_context data dict, or None if not found.
    """
    for ctx in structured_event.context:
        if ctx.schema == InternalEventsClient.AI_CONTEXT_SCHEMA:
            return ctx.data
    return None


def test_ai_context_schema_version():
    """Verify AI_CONTEXT_SCHEMA is the expected version with cache_creation support."""
    assert (
        InternalEventsClient.AI_CONTEXT_SCHEMA
        == "iglu:com.gitlab/ai_context/jsonschema/1-0-1"
    )


class TestInternalEventsClientAIContext:
    """Test InternalEventsClient AIContext extraction from extra."""

    @pytest.fixture
    def mock_tracker(self):
        """Create a mock Snowplow tracker."""
        return Mock()

    @pytest.fixture
    def client(self, mock_tracker):
        """Create a client with mocked dependencies."""
        with (
            patch("lib.internal_events.client.requests.Session"),
            patch("lib.internal_events.client.AsyncEmitter"),
            patch("lib.internal_events.client.Tracker") as tracker_class,
        ):
            tracker_class.return_value = mock_tracker
            client = InternalEventsClient(
                enabled=True,
                endpoint="https://test.endpoint.com",
                app_id="test_app",
                namespace="test_namespace",
                batch_size=1,
                thread_count=1,
            )
            yield client

    @pytest.mark.parametrize(
        "cache_creation_value,expected_cache_creation",
        [
            pytest.param(20, 20, id="extracts_cache_creation_value"),
            pytest.param(None, None, id="handles_missing_cache_creation"),
            pytest.param(0, 0, id="handles_zero_cache_creation"),
        ],
    )
    def test_track_event_cache_creation_extraction(
        self, client, mock_tracker, cache_creation_value, expected_cache_creation
    ):
        """Verify cache_creation is correctly extracted from extra to AIContext."""
        current_event_context.set(EventContext())

        additional_properties_kwargs = {
            "label": "cache_details",
            "cache_read": 10,
        }
        if cache_creation_value is not None:
            additional_properties_kwargs["cache_creation"] = cache_creation_value

        additional_properties = InternalEventAdditionalProperties(
            **additional_properties_kwargs
        )

        client.track_event(
            "test_token_usage_event",
            additional_properties=additional_properties,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        mock_tracker.track.assert_called_once()
        call_args = mock_tracker.track.call_args
        structured_event = call_args[0][0]

        ai_context_json = _find_ai_context(structured_event)
        assert ai_context_json is not None, "AIContext not found in structured event"
        assert ai_context_json["cache_creation"] == expected_cache_creation
        assert ai_context_json["cache_read"] == 10

    def test_track_event_extracts_all_cache_fields_to_ai_context(
        self, client, mock_tracker
    ):
        """Verify all cache-related fields are extracted from extra to AIContext."""
        current_event_context.set(EventContext())

        additional_properties = InternalEventAdditionalProperties(
            label="cache_details",
            cache_read=10,
            cache_creation=20,
            ephemeral_5m_input_tokens=5,
            ephemeral_1h_input_tokens=15,
        )

        client.track_event(
            "test_token_usage_event",
            additional_properties=additional_properties,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        mock_tracker.track.assert_called_once()
        call_args = mock_tracker.track.call_args
        structured_event = call_args[0][0]

        ai_context_json = _find_ai_context(structured_event)
        assert ai_context_json is not None, "AIContext not found in structured event"
        assert ai_context_json["cache_creation"] == 20
        assert ai_context_json["cache_read"] == 10
        assert ai_context_json["ephemeral_5m_input_tokens"] == 5
        assert ai_context_json["ephemeral_1h_input_tokens"] == 15
