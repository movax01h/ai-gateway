"""Tests for InternalEventsClient AIContext extraction."""

from unittest.mock import Mock, patch

import pytest

from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
    merge_request_url_context,
    pipeline_source_context,
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

    def test_track_event_forwards_context_extra_into_emitted_payload(
        self, client, mock_tracker
    ):
        """The request-scoped EventContext.extra (e.g. the client-supplied x-gitlab-tracking-context fields parsed by
        the middleware/interceptor) must reach the emitted gitlab_standard payload, merged with the per-event extras —
        per-event keys win on conflict."""
        current_event_context.set(
            EventContext(
                extra={
                    "distribution": "glab",
                    "execution_environment": "gitlab_ci",
                    "lsp_version": "9.6.0",
                    "shared_key": "from_context",
                }
            )
        )
        additional_properties = InternalEventAdditionalProperties(
            label="test-label",
            workflow_id="123",
            shared_key="from_event",
        )

        client.track_event("some_event", additional_properties=additional_properties)

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        standard_context = next(
            ctx
            for ctx in structured_event.context
            if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
        )
        assert standard_context.data["extra"] == {
            "distribution": "glab",
            "execution_environment": "gitlab_ci",
            "lsp_version": "9.6.0",
            "workflow_id": "123",
            "shared_key": "from_event",
        }
        # The caller's dict must not be mutated by the merge.
        assert additional_properties.extra == {
            "workflow_id": "123",
            "shared_key": "from_event",
        }

    def test_track_event_includes_merge_request_url_when_context_set(
        self, client, mock_tracker
    ):
        """Verify merge_request_url is injected into extra when ContextVar is set."""
        current_event_context.set(EventContext())
        token = merge_request_url_context.set(
            "https://gitlab.com/group/project/-/merge_requests/1"
        )
        try:
            client.track_event(
                "test_event",
                additional_properties=InternalEventAdditionalProperties(
                    label="test_label",
                ),
            )

            mock_tracker.track.assert_called_once()
            structured_event = mock_tracker.track.call_args[0][0]
            standard_context = next(
                ctx
                for ctx in structured_event.context
                if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
            )
            assert (
                standard_context.data["extra"]["merge_request_url"]
                == "https://gitlab.com/group/project/-/merge_requests/1"
            )
        finally:
            merge_request_url_context.reset(token)

    def test_track_event_omits_merge_request_url_when_context_not_set(
        self, client, mock_tracker
    ):
        """Verify merge_request_url is not in extra when ContextVar is unset."""
        current_event_context.set(EventContext())
        token = merge_request_url_context.set(None)
        try:
            client.track_event(
                "test_event",
                additional_properties=InternalEventAdditionalProperties(
                    label="test_label",
                ),
            )

            mock_tracker.track.assert_called_once()
            structured_event = mock_tracker.track.call_args[0][0]
            standard_context = next(
                ctx
                for ctx in structured_event.context
                if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
            )
            assert "merge_request_url" not in standard_context.data["extra"]
        finally:
            merge_request_url_context.reset(token)

    def test_track_event_includes_pipeline_source_when_context_set(
        self, client, mock_tracker
    ):
        """Verify pipeline_source is injected into extra when ContextVar is set."""
        current_event_context.set(EventContext())
        token = pipeline_source_context.set("merge_request_event")
        try:
            client.track_event(
                "test_event",
                additional_properties=InternalEventAdditionalProperties(
                    label="test_label",
                ),
            )

            mock_tracker.track.assert_called_once()
            structured_event = mock_tracker.track.call_args[0][0]
            standard_context = next(
                ctx
                for ctx in structured_event.context
                if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
            )
            assert (
                standard_context.data["extra"]["pipeline_source"]
                == "merge_request_event"
            )
        finally:
            pipeline_source_context.reset(token)

    def test_track_event_omits_pipeline_source_when_context_not_set(
        self, client, mock_tracker
    ):
        """Verify pipeline_source is not in extra when ContextVar is unset."""
        current_event_context.set(EventContext())
        token = pipeline_source_context.set(None)
        try:
            client.track_event(
                "test_event",
                additional_properties=InternalEventAdditionalProperties(
                    label="test_label",
                ),
            )

            mock_tracker.track.assert_called_once()
            structured_event = mock_tracker.track.call_args[0][0]
            standard_context = next(
                ctx
                for ctx in structured_event.context
                if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
            )
            assert "pipeline_source" not in standard_context.data["extra"]
        finally:
            pipeline_source_context.reset(token)


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


class TestInternalEventsClientExplicitAIContext:
    """Test InternalEventsClient with explicit AIContext parameter."""

    @pytest.fixture
    def mock_tracker(self):
        return Mock()

    @pytest.fixture
    def client(self, mock_tracker):
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

    def _find_ai_context(self, structured_event):
        for ctx in structured_event.context:
            if ctx.schema == InternalEventsClient.AI_CONTEXT_SCHEMA:
                return ctx.data
        return None

    def test_explicit_ai_context_workflow_fields_take_precedence_over_extra(
        self, client, mock_tracker
    ):
        """Explicit AIContext workflow_id/flow_type/agent_name override values in extra."""
        from lib.internal_events.ai_context import AIContext

        current_event_context.set(EventContext())

        explicit_ctx = AIContext(
            workflow_id="explicit-wf-id",
            flow_type="explicit_flow",
            agent_name="explicit_agent",
        )
        additional_properties = InternalEventAdditionalProperties(
            label="test",
            workflow_id="implicit-wf-id",
            workflow_type="implicit_flow",
            agent_name="implicit_agent",
        )

        client.track_event(
            "test_event",
            additional_properties=additional_properties,
            ai_context=explicit_ctx,
        )

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        ai_ctx_data = self._find_ai_context(structured_event)

        assert ai_ctx_data is not None
        assert ai_ctx_data["workflow_id"] == "explicit-wf-id"
        assert ai_ctx_data["flow_type"] == "explicit_flow"
        assert ai_ctx_data["agent_name"] == "explicit_agent"

    def test_explicit_ai_context_token_fields_take_precedence_over_kwargs(
        self, client, mock_tracker
    ):
        """Explicit AIContext token fields override values from kwargs."""
        from lib.internal_events.ai_context import AIContext

        current_event_context.set(EventContext())

        explicit_ctx = AIContext(
            input_tokens=999,
            output_tokens=888,
            total_tokens=1887,
            cache_read=77,
            cache_creation=66,
            ephemeral_5m_input_tokens=55,
            ephemeral_1h_input_tokens=44,
        )

        client.track_event(
            "test_event",
            ai_context=explicit_ctx,
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
        )

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        ai_ctx_data = self._find_ai_context(structured_event)

        assert ai_ctx_data is not None
        assert ai_ctx_data["input_tokens"] == 999
        assert ai_ctx_data["output_tokens"] == 888
        assert ai_ctx_data["total_tokens"] == 1887
        assert ai_ctx_data["cache_read"] == 77
        assert ai_ctx_data["cache_creation"] == 66
        assert ai_ctx_data["ephemeral_5m_input_tokens"] == 55
        assert ai_ctx_data["ephemeral_1h_input_tokens"] == 44

    def test_explicit_ai_context_none_falls_back_to_implicit_extraction(
        self, client, mock_tracker
    ):
        """When ai_context=None, implicit extraction from extra/kwargs still works."""
        current_event_context.set(EventContext())

        additional_properties = InternalEventAdditionalProperties(
            label="test",
            workflow_id="implicit-wf-id",
            workflow_type="implicit_flow",
            agent_name="implicit_agent",
            cache_read=10,
            cache_creation=20,
        )

        client.track_event(
            "test_event",
            additional_properties=additional_properties,
            ai_context=None,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        ai_ctx_data = self._find_ai_context(structured_event)

        assert ai_ctx_data is not None
        assert ai_ctx_data["workflow_id"] == "implicit-wf-id"
        assert ai_ctx_data["flow_type"] == "implicit_flow"
        assert ai_ctx_data["agent_name"] == "implicit_agent"
        assert ai_ctx_data["input_tokens"] == 100
        assert ai_ctx_data["output_tokens"] == 50
        assert ai_ctx_data["total_tokens"] == 150
        assert ai_ctx_data["cache_read"] == 10
        assert ai_ctx_data["cache_creation"] == 20

    def test_explicit_ai_context_session_id_from_additional_properties_value(
        self, client, mock_tracker
    ):
        """session_id in AIContext is always derived from additional_properties.value."""
        from lib.internal_events.ai_context import AIContext

        current_event_context.set(EventContext())

        explicit_ctx = AIContext(workflow_id="wf-123")
        additional_properties = InternalEventAdditionalProperties(
            label="test",
            value=42,
        )

        client.track_event(
            "test_event",
            additional_properties=additional_properties,
            ai_context=explicit_ctx,
        )

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        ai_ctx_data = self._find_ai_context(structured_event)

        assert ai_ctx_data is not None
        assert ai_ctx_data["session_id"] == "42"
        assert ai_ctx_data["workflow_id"] == "wf-123"

    def test_explicit_ai_context_partial_fields_use_explicit_for_set_fields(
        self, client, mock_tracker
    ):
        """Explicit AIContext with only some fields set: set fields override implicit,
        None fields on explicit context still override implicit (explicit wins entirely)."""
        from lib.internal_events.ai_context import AIContext

        current_event_context.set(EventContext())

        # Only workflow_id is set on explicit context; flow_type and agent_name are None
        explicit_ctx = AIContext(workflow_id="explicit-wf")
        additional_properties = InternalEventAdditionalProperties(
            label="test",
            workflow_id="implicit-wf",
            workflow_type="implicit_flow",
            agent_name="implicit_agent",
        )

        client.track_event(
            "test_event",
            additional_properties=additional_properties,
            ai_context=explicit_ctx,
        )

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        ai_ctx_data = self._find_ai_context(structured_event)

        assert ai_ctx_data is not None
        # Explicit context is used entirely when provided; None fields on it stay None
        assert ai_ctx_data["workflow_id"] == "explicit-wf"
        assert ai_ctx_data["flow_type"] is None
        assert ai_ctx_data["agent_name"] is None

    def test_explicit_ai_context_standard_context_payload_unchanged(
        self, client, mock_tracker
    ):
        """Passing explicit ai_context does not alter the gitlab_standard payload."""
        from lib.internal_events.ai_context import AIContext

        current_event_context.set(EventContext())

        explicit_ctx = AIContext(workflow_id="wf-abc", flow_type="chat")
        additional_properties = InternalEventAdditionalProperties(
            label="test",
            workflow_id="wf-abc",
        )

        client.track_event(
            "test_event",
            additional_properties=additional_properties,
            ai_context=explicit_ctx,
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
        )

        mock_tracker.track.assert_called_once()
        structured_event = mock_tracker.track.call_args[0][0]
        standard_ctx = next(
            ctx
            for ctx in structured_event.context
            if ctx.schema == InternalEventsClient.STANDARD_CONTEXT_SCHEMA
        )
        # Token values still appear in the standard context (backwards compat)
        assert standard_ctx.data["input_tokens"] == 5
        assert standard_ctx.data["output_tokens"] == 10
        assert standard_ctx.data["total_tokens"] == 15
        # workflow_id still appears in extra (backwards compat)
        assert standard_ctx.data["extra"]["workflow_id"] == "wf-abc"
