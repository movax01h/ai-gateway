from unittest.mock import MagicMock, Mock

import pytest

from duo_workflow_service.audit_events.collector import AuditEventCollector
from duo_workflow_service.audit_events.context import audit_collector_context
from duo_workflow_service.audit_events.web_tools import capture_native_web_calls
from duo_workflow_service.tracking.monitoring_context import (
    MonitoringContext,
    current_monitoring_context,
)
from lib.context import current_model_metadata_context
from lib.internal_events import InternalEventsClient
from lib.internal_events.event_enum import EventEnum

SEARCH_BLOCK = {
    "type": "server_tool_use",
    "id": "srvtu_1",
    "name": "web_search",
    "input": {"query": "gitlab duo"},
}
FETCH_BLOCK = {
    "type": "server_tool_use",
    "id": "srvtu_2",
    "name": "web_fetch",
    "input": {"url": "https://example.com"},
}


@pytest.fixture(name="internal_event_client")
def internal_event_client_fixture(mock_duo_workflow_service_container):
    client = MagicMock(spec=InternalEventsClient)
    with mock_duo_workflow_service_container.internal_event.client.override(client):
        yield client


@pytest.fixture(autouse=True)
def session_context():
    monitoring_token = current_monitoring_context.set(
        MonitoringContext(workflow_id="42", workflow_definition="chat")
    )
    metadata = Mock()
    metadata.llm_definition.provider = "Anthropic"
    metadata.llm_definition.params.model = "claude-sonnet-4-6"
    model_token = current_model_metadata_context.set(metadata)
    try:
        yield
    finally:
        current_model_metadata_context.reset(model_token)
        current_monitoring_context.reset(monitoring_token)


def _tracked(client):
    return [
        call.kwargs
        for call in client.track_event.call_args_list
        if call.kwargs["event_name"] == EventEnum.WEB_TOOL_INVOKED.value
    ]


class TestNativeWebToolUsageEvent:
    @pytest.mark.parametrize(
        ("block", "tool_name"),
        [
            pytest.param(SEARCH_BLOCK, "web_search", id="search"),
            pytest.param(FETCH_BLOCK, "web_fetch", id="fetch"),
        ],
    )
    def test_tracks_each_native_call_with_model_and_source(
        self, internal_event_client, block, tool_name
    ):
        capture_native_web_calls([block], "claude-haiku-4-5")

        [event] = _tracked(internal_event_client)
        properties = event["additional_properties"]
        assert (properties.label, properties.property, properties.value) == (
            "native",
            tool_name,
            "42",
        )
        assert (event["model_name"], event["model_provider"]) == (
            "claude-haiku-4-5",
            "Anthropic",
        )
        assert event["category"] == "chat"

    @pytest.mark.parametrize(
        ("content", "status"),
        [
            pytest.param(
                [
                    SEARCH_BLOCK,
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtu_1",
                        "content": [{"type": "web_search_result", "url": "https://a"}],
                    },
                ],
                "success",
                id="anthropic-results",
            ),
            pytest.param(
                [
                    SEARCH_BLOCK,
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtu_1",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "max_uses_exceeded",
                        },
                    },
                ],
                "failure",
                id="anthropic-error",
            ),
            pytest.param(
                [
                    FETCH_BLOCK,
                    {
                        "type": "web_fetch_tool_result",
                        "tool_use_id": "srvtu_2",
                        "content": {
                            "type": "web_fetch_tool_result_error",
                            "error_code": "url_not_accessible",
                        },
                    },
                ],
                "failure",
                id="anthropic-fetch-error",
            ),
            pytest.param(
                [
                    SEARCH_BLOCK,
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtu_1",
                        "content": [],
                    },
                ],
                "success",
                id="anthropic-zero-results",
            ),
            pytest.param([SEARCH_BLOCK], "failure", id="anthropic-no-result"),
            pytest.param(
                [
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {"type": "search"},
                    }
                ],
                "success",
                id="openai-completed",
            ),
            pytest.param(
                [
                    {
                        "type": "web_search_call",
                        "status": "failed",
                        "action": {"type": "search"},
                    }
                ],
                "failure",
                id="openai-failed",
            ),
        ],
    )
    def test_tracks_whether_the_call_succeeded(
        self, internal_event_client, content, status
    ):
        capture_native_web_calls(content)

        [event] = _tracked(internal_event_client)
        assert event["additional_properties"].extra["status"] == status

    def test_tracks_without_an_audit_collector(self, internal_event_client):
        assert audit_collector_context.get() is None

        capture_native_web_calls([SEARCH_BLOCK, FETCH_BLOCK])

        assert [
            e["additional_properties"].property for e in _tracked(internal_event_client)
        ] == ["web_search", "web_fetch"]

    def test_tracks_the_same_calls_the_audit_log_records(self, internal_event_client):
        """OpenAI find_in_page on an already-read URL is no new fetch, for either sink."""

        def call(action):
            return {"type": "web_search_call", "status": "completed", "action": action}

        blocks = [
            call({"type": "open_page", "url": "https://x"}),
            call({"type": "find_in_page", "url": "https://x", "pattern": "CVE"}),
            call({"type": "find_in_page", "url": "https://y", "pattern": "CVE"}),
        ]
        collector = AuditEventCollector(
            client=Mock(), workflow_id="42", flush_interval_seconds=1_000
        )
        token = audit_collector_context.set(collector)
        try:
            capture_native_web_calls(blocks)
        finally:
            audit_collector_context.reset(token)

        audited = collector._buffer  # pylint: disable=protected-access
        assert len(_tracked(internal_event_client)) == len(audited) == 2

    def test_ignores_the_client_executed_fallback_tool(self, internal_event_client):
        """AgentCore searches are tracked by the tool itself."""
        capture_native_web_calls([{**SEARCH_BLOCK, "type": "tool_use"}])

        assert _tracked(internal_event_client) == []

    def test_never_sends_the_query_or_url(self, internal_event_client):
        capture_native_web_calls([SEARCH_BLOCK, FETCH_BLOCK])

        sent = str(_tracked(internal_event_client))
        assert "gitlab duo" not in sent
        assert "example.com" not in sent
