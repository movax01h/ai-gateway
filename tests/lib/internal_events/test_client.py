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
            patch("lib.internal_events.client.LoggingAsyncEmitter"),
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

    def test_track_event_standard_context_field_routing(self, client, mock_tracker):
        """The gitlab_standard payload defines `user_type` but not `subject_type`.

        `user_type` must be present (sourced from EventContext); `subject_type`
        must be absent (it is billing-only and lives on a dedicated ContextVar).
        """
        current_event_context.set(EventContext(user_type="service_account"))

        client.track_event("some_event")

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]

        standard_context = next(
            ctx
            for ctx in structured_event.context
            if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
        )
        assert standard_context.data["user_type"] == "service_account"
        assert "subject_type" not in standard_context.data


class TestTruncateString:
    """Test InternalEventsClient.truncate_string length validator."""

    @pytest.fixture
    def client(self):
        with (
            patch("lib.internal_events.client.requests.Session"),
            patch("lib.internal_events.client.LoggingAsyncEmitter"),
            patch("lib.internal_events.client.Tracker"),
        ):
            yield InternalEventsClient(
                enabled=True,
                endpoint="https://test.endpoint.com",
                app_id="test_app",
                namespace="test_namespace",
                batch_size=1,
                thread_count=1,
            )

    @pytest.mark.parametrize(
        "value,expected",
        [
            pytest.param(None, None, id="none_returns_none"),
            pytest.param("", "", id="empty_string_returns_empty"),
            pytest.param("short value", "short value", id="short_string_unchanged"),
            pytest.param(
                "a" * InternalEventsClient.MAX_VALUE_LENGTH,
                "a" * InternalEventsClient.MAX_VALUE_LENGTH,
                id="exactly_max_length_unchanged",
            ),
        ],
    )
    def test_truncate_string_does_not_truncate(self, client, value, expected):
        assert client.truncate_string(value) == expected

    def test_truncate_string_truncates_when_over_max(self, client):
        value = "a" * (InternalEventsClient.MAX_VALUE_LENGTH + 50)

        result = client.truncate_string(value)

        assert result is not None
        assert len(result) == InternalEventsClient.MAX_VALUE_LENGTH
        assert result.endswith("...")
        assert result == "a" * (InternalEventsClient.MAX_VALUE_LENGTH - 3) + "..."

    def test_truncate_string_preserves_prefix_content(self, client):
        prefix = "important-prefix:"
        value = prefix + "x" * InternalEventsClient.MAX_VALUE_LENGTH

        result = client.truncate_string(value)

        assert result is not None
        assert result.startswith(prefix)
        assert result.endswith("...")
        assert len(result) == InternalEventsClient.MAX_VALUE_LENGTH
