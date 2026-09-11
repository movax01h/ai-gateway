"""Tests for UnifiedEventService."""

import uuid
from datetime import datetime
from unittest.mock import Mock

import pytest
import requests

from lib.internal_events.context import (
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
    merge_request_url_context,
    pipeline_source_context,
    tracked_internal_events,
)
from lib.unified_events.service import UnifiedEventService


class InlineExecutor:
    """Runs submitted work synchronously so tests can assert on the POST."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


def _find_context(body, schema):
    for ctx in body["contexts"]:
        if ctx["schema"] == schema:
            return ctx["data"]
    return None


@pytest.fixture
def mock_session():
    session = Mock()
    session.post.return_value = Mock(status_code=202, text="")
    return session


@pytest.fixture
def service(mock_session):
    service = UnifiedEventService()
    service._session = mock_session
    service._executor = InlineExecutor()
    return service


@pytest.fixture
def posted_body(mock_session):
    def _posted_body():
        return mock_session.post.call_args.kwargs["json"]

    return _posted_body


@pytest.fixture
def event_context():
    context = EventContext(
        environment="production",
        source="ai-gateway-python",
        realm="saas",
        deployment_type=".com",
        unique_instance_id="instance-uuid",
        host_name="gitlab.example.com",
        instance_version="18.6.0",
        global_user_id="global-user-id",
        user_id="hashed-user-id",
        is_gitlab_team_member=False,
        project_id=1,
        namespace_id=2,
        ultimate_parent_namespace_id=3,
        organization_id=4,
        plan="ultimate",
        correlation_id="corr-id",
        feature_enablement_type="duo_pro",
    )
    token = current_event_context.set(context)
    yield context
    current_event_context.reset(token)


@pytest.fixture
def tracked_events():
    token = tracked_internal_events.set(set())
    yield tracked_internal_events.get()
    tracked_internal_events.reset(token)


class TestUnifiedEventServiceTrack:
    def test_posts_single_event_to_sidecar(
        self, service, mock_session, posted_body, event_context, tracked_events
    ):
        service.track("request_duo_chat")

        assert mock_session.post.call_count == 1
        url = mock_session.post.call_args.args[0]
        assert url == f"{UnifiedEventService.SIDECAR_URL}/v1/events"
        assert (
            mock_session.post.call_args.kwargs["timeout"]
            == UnifiedEventService.REQUEST_TIMEOUT
        )
        schemas = [ctx["schema"] for ctx in posted_body()["contexts"]]
        assert schemas == [UnifiedEventService.AI_CONTEXT_SCHEMA]

    def test_event_construct_envelope(
        self, service, posted_body, event_context, tracked_events
    ):
        service.track("request_duo_chat")

        body = posted_body()
        assert uuid.UUID(body["event_id"]).version == 4
        assert body["event_type"] == "request_duo_chat"
        assert body["timestamp"].endswith("Z")
        datetime.fromisoformat(body["timestamp"].replace("Z", "+00:00"))

    def test_event_construct_is_populated_from_event_context(
        self, service, posted_body, event_context, tracked_events
    ):
        service.track("request_duo_chat")

        body = posted_body()
        assert body["environment"] == "production"
        assert body["source"] == "ai-gateway-python"
        assert body["realm"] == "saas"
        assert body["deployment_type"] == ".com"
        assert body["unique_instance_id"] == "instance-uuid"
        assert body["host_name"] == "gitlab.example.com"
        assert body["instance_version"] == "18.6.0"
        assert body["global_user_id"] == "global-user-id"
        assert body["user_id"] == "hashed-user-id"
        assert body["is_gitlab_team_member"] is False
        assert body["project_id"] == 1
        assert body["namespace_id"] == 2
        assert body["root_namespace_id"] == 3
        assert body["organization_id"] == 4
        assert body["plan"] == "ultimate"
        assert body["correlation_id"] == "corr-id"
        assert body["feature_enablement_type"] == "duo_pro"

    def test_structured_event_fields_are_top_level(
        self, service, posted_body, event_context, tracked_events
    ):
        additional_properties = InternalEventAdditionalProperties(
            label="unit_primitive", property="prop", value=7, workflow_id="wf-1"
        )

        service.track(
            "request_duo_chat",
            additional_properties=additional_properties,
            category="MyClass",
        )

        body = posted_body()
        assert body["category"] == "MyClass"
        assert body["label"] == "unit_primitive"
        assert body["property"] == "prop"
        assert body["value"] == 7
        assert body["metadata"] == {"workflow_id": "wf-1"}

    def test_category_falls_back_to_default(
        self, service, posted_body, event_context, tracked_events
    ):
        service.track("request_duo_chat", category=None)

        body = posted_body()
        assert body["category"] == "default_category"
        assert "label" not in body
        assert "property" not in body
        assert "value" not in body

    def test_property_is_truncated_to_sidecar_limit(
        self, service, posted_body, event_context, tracked_events
    ):
        additional_properties = InternalEventAdditionalProperties(property="x" * 2000)

        service.track("request_duo_chat", additional_properties=additional_properties)

        assert len(posted_body()["property"]) == 255

    def test_ai_context_and_metadata_are_populated(
        self, service, posted_body, event_context, tracked_events
    ):
        additional_properties = InternalEventAdditionalProperties(
            label="unit_primitive",
            property="prop",
            value=7,
            workflow_id="wf-1",
            agent_name="agent",
        )
        mr_token = merge_request_url_context.set("https://gitlab.com/mr/1")
        pipeline_token = pipeline_source_context.set("web")
        try:
            service.track(
                "request_duo_chat",
                additional_properties=additional_properties,
                input_tokens=10,
            )
        finally:
            merge_request_url_context.reset(mr_token)
            pipeline_source_context.reset(pipeline_token)

        body = posted_body()
        assert body["metadata"] == {
            "workflow_id": "wf-1",
            "agent_name": "agent",
            "merge_request_url": "https://gitlab.com/mr/1",
            "pipeline_source": "web",
        }

        ai_context = _find_context(body, UnifiedEventService.AI_CONTEXT_SCHEMA)
        assert ai_context["session_id"] == "7"
        assert ai_context["workflow_id"] == "wf-1"
        assert ai_context["agent_name"] == "agent"
        assert ai_context["input_tokens"] == 10

    def test_non_token_kwargs_land_in_metadata(
        self, service, posted_body, event_context, tracked_events
    ):
        service.track(
            "token_usage_duo_chat",
            model_name="claude-sonnet-5",
            model_provider="anthropic",
            input_tokens=10,
        )

        body = posted_body()
        assert body["metadata"] == {
            "model_name": "claude-sonnet-5",
            "model_provider": "anthropic",
        }
        ai_context = _find_context(body, UnifiedEventService.AI_CONTEXT_SCHEMA)
        assert ai_context["input_tokens"] == 10

    def test_callers_extra_is_not_mutated(self, service, event_context, tracked_events):
        additional_properties = InternalEventAdditionalProperties(workflow_id="wf-1")
        mr_token = merge_request_url_context.set("https://gitlab.com/mr/1")
        try:
            service.track(
                "request_duo_chat", additional_properties=additional_properties
            )
        finally:
            merge_request_url_context.reset(mr_token)

        assert additional_properties.extra == {"workflow_id": "wf-1"}

    def test_context_extra_is_merged_and_event_extra_wins(
        self, service, posted_body, tracked_events
    ):
        context = EventContext(
            extra={"lsp_version": "4.1.0", "workflow_id": "from-context"}
        )
        token = current_event_context.set(context)
        additional_properties = InternalEventAdditionalProperties(workflow_id="wf-1")
        try:
            service.track(
                "request_duo_chat", additional_properties=additional_properties
            )
        finally:
            current_event_context.reset(token)

        assert posted_body()["metadata"] == {
            "lsp_version": "4.1.0",
            "workflow_id": "wf-1",
        }

    def test_records_tracked_event_name(self, service, event_context, tracked_events):
        service.track("request_duo_chat")

        assert "request_duo_chat" in tracked_events

    def test_invalid_construct_field_drops_event(
        self, service, mock_session, tracked_events
    ):
        context = EventContext(realm="not-a-realm")
        token = current_event_context.set(context)
        try:
            service.track("request_duo_chat")
        finally:
            current_event_context.reset(token)

        mock_session.post.assert_not_called()
        assert "request_duo_chat" not in tracked_events

    def test_rejected_event_is_logged_not_raised(
        self, service, mock_session, event_context, tracked_events
    ):
        mock_session.post.return_value = Mock(
            status_code=400, text='{"error": "invalid_field", "field": "realm"}'
        )

        service.track("request_duo_chat")

        assert mock_session.post.call_count == 1

    def test_request_exception_is_logged_not_raised(
        self, service, mock_session, event_context, tracked_events
    ):
        mock_session.post.side_effect = requests.ConnectionError("sidecar down")

        service.track("request_duo_chat")

        assert mock_session.post.call_count == 1
