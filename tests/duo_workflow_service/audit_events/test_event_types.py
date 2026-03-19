import uuid
from datetime import datetime, timezone

import pytest

from duo_workflow_service.audit_events.event_types import (
    CLOUDEVENT_DATA_CONTENT_TYPE,
    CLOUDEVENT_SOURCE,
    CLOUDEVENT_SPEC_VERSION,
    LlmInputSentEvent,
    LlmResponseReceivedEvent,
    SessionEndedEvent,
    SessionStartedEvent,
    ToolExecutionFailedEvent,
    ToolExecutionRetriedEvent,
    ToolInvokedEvent,
    ToolResponseReceivedEvent,
    UserInputReceivedEvent,
    UserOutputDisplayedEvent,
)


class TestAuditEventBase:
    def test_auto_generates_uuid(self):
        event = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        uuid.UUID(event.id)

    def test_unique_ids(self):
        event1 = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        event2 = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        assert event1.id != event2.id

    def test_auto_generates_timestamp(self):
        before = datetime.now(timezone.utc)
        event = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_explicit_id_preserved(self):
        event = ToolInvokedEvent(
            id="custom-id", workflow_id="wf-1", tool_name="read_file"
        )
        assert event.id == "custom-id"


class TestCloudEventSerialization:
    def test_required_attributes_present(self):
        event = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        ce = event.to_cloudevent()
        assert ce["specversion"] == CLOUDEVENT_SPEC_VERSION
        assert ce["id"] == event.id
        assert ce["type"] == "ai_tool_invoked"
        assert ce["source"] == CLOUDEVENT_SOURCE

    def test_optional_attributes_present(self):
        event = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        ce = event.to_cloudevent()
        assert ce["time"] == event.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        assert ce["subject"] == "wf-1"
        assert ce["datacontenttype"] == CLOUDEVENT_DATA_CONTENT_TYPE

    def test_data_contains_domain_fields(self):
        event = ToolInvokedEvent(
            workflow_id="wf-1",
            tool_name="read_file",
            tool_args={"path": "/src/main.py"},
        )
        ce = event.to_cloudevent()
        assert ce["data"]["tool_name"] == "read_file"
        assert ce["data"]["tool_args"] == {"path": "/src/main.py"}
        assert ce["data"]["workflow_id"] == "wf-1"

    def test_data_excludes_envelope_fields(self):
        event = ToolInvokedEvent(workflow_id="wf-1", tool_name="read_file")
        ce = event.to_cloudevent()
        assert "id" not in ce["data"]
        assert "event_type" not in ce["data"]
        assert "timestamp" not in ce["data"]

    @pytest.mark.parametrize(
        "event_class,event_type_value,kwargs",
        [
            (
                SessionStartedEvent,
                "ai_agent_session_started",
                {"workflow_type": "issue_to_mr", "goal": "Fix bug"},
            ),
            (
                SessionEndedEvent,
                "ai_agent_session_ended",
                {"status": "success"},
            ),
            (
                UserInputReceivedEvent,
                "ai_user_input_received",
                {"input_type": "text", "content": "hello", "content_length": 5},
            ),
            (
                LlmInputSentEvent,
                "ai_llm_input_sent",
                {"model_name": "claude-3", "prompt_content": "test prompt"},
            ),
            (
                LlmResponseReceivedEvent,
                "ai_llm_response_received",
                {"model_name": "claude-3", "response_content": "test response"},
            ),
            (
                UserOutputDisplayedEvent,
                "ai_user_output_displayed",
                {"output_type": "text", "content": "result", "content_length": 6},
            ),
            (
                ToolInvokedEvent,
                "ai_tool_invoked",
                {"tool_name": "read_file"},
            ),
            (
                ToolResponseReceivedEvent,
                "ai_tool_response_received",
                {
                    "tool_name": "read_file",
                    "response_content": "data",
                    "response_length": 4,
                },
            ),
            (
                ToolExecutionFailedEvent,
                "ai_tool_execution_failed",
                {
                    "tool_name": "read_file",
                    "error_type": "IOError",
                    "error_message": "not found",
                },
            ),
            (
                ToolExecutionRetriedEvent,
                "ai_tool_execution_retried",
                {"tool_name": "read_file", "attempt_number": 2, "max_attempts": 3},
            ),
        ],
    )
    def test_cloudevent_type_mapping(self, event_class, event_type_value, kwargs):
        event = event_class(workflow_id="wf-1", **kwargs)
        ce = event.to_cloudevent()
        assert ce["type"] == event_type_value

    def test_optional_fields_in_data(self):
        event = LlmResponseReceivedEvent(
            workflow_id="wf-1",
            model_name="claude-3",
            response_content="hello",
            prompt_token_count=100,
            completion_token_count=50,
            finish_reason="end_turn",
            latency_ms=1234.5,
        )
        ce = event.to_cloudevent()
        assert ce["data"]["prompt_token_count"] == 100
        assert ce["data"]["completion_token_count"] == 50
        assert ce["data"]["finish_reason"] == "end_turn"
        assert ce["data"]["latency_ms"] == 1234.5

    def test_none_optional_fields_in_data(self):
        event = LlmResponseReceivedEvent(
            workflow_id="wf-1",
            model_name="claude-3",
            response_content="hello",
        )
        ce = event.to_cloudevent()
        assert ce["data"]["prompt_token_count"] is None
        assert ce["data"]["completion_token_count"] is None
        assert ce["data"]["finish_reason"] is None
        assert ce["data"]["latency_ms"] is None
