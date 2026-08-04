from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import BaseTool, ToolException
from pydantic_core import ValidationError

from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from lib.internal_events.event_enum import EventEnum


@pytest.fixture(name="tracker")
def tracker_fixture():
    """ToolEventTracker with mocked dependencies."""
    with (
        patch(
            "duo_workflow_service.agent_platform.utils.tool_event_tracker.duo_workflow_metrics"
        ) as mock_metrics,
        patch(
            "duo_workflow_service.agent_platform.utils.tool_event_tracker.client_capabilities"
        ) as mock_capabilities,
        patch(
            "duo_workflow_service.agent_platform.utils.tool_event_tracker.tool_executions"
        ) as mock_executions,
    ):
        mock_capabilities.get.return_value = {"cap1", "cap2"}
        mock_executions.get.return_value = []

        internal_client = Mock()
        tracker = ToolEventTracker(
            flow_id="test-flow-123",
            flow_type=Mock(value="software_development"),
            internal_event_client=internal_client,
        )
        yield SimpleNamespace(
            tracker=tracker,
            metrics=mock_metrics,
            client=internal_client,
            executions=mock_executions,
            capabilities=mock_capabilities,
        )


class TestTrackInternalEvent:
    def test_tracks_success_event(self, tracker):
        tracker.tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
            tool_name="gitlab_search",
        )

        tracker.client.track_event.assert_called_once()
        call_kwargs = tracker.client.track_event.call_args.kwargs
        additional_properties = call_kwargs["additional_properties"]
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_TOOL_SUCCESS.value
        assert call_kwargs["category"] == "software_development"
        assert additional_properties.property == "gitlab_search"
        assert additional_properties.value == "test-flow-123"
        assert set(additional_properties.extra["client_capabilities"]) == {
            "cap1",
            "cap2",
        }

    def test_appends_tool_name_on_success(self, tracker):

        mock_list = []

        tracker.executions.get.return_value = mock_list

        tracker.tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
            tool_name="gitlab_search",
        )

        assert "gitlab_search" in mock_list

    def test_does_not_append_when_tool_executions_is_none(self, tracker):

        tracker.executions.get.return_value = None  # context var unavailable
        # Should not raise
        tracker.tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
            tool_name="gitlab_search",
        )
        tracker.client.track_event.assert_called_once()

    def test_handles_empty_tool_name(self, tracker):
        tracker.tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name="",
            extra={"error_type": "SomeError"},
        )
        # _record_metric falls back to "unknown"
        tracker.metrics.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type="software_development",
            tool_name="unknown",
            failure_reason="SomeError",
        )

    def test_handles_none_extra(self, tracker):
        tracker.tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name="my_tool",
            extra=None,  # default
        )
        # error_type missing -> "unknown" failure_reason
        tracker.metrics.count_agent_platform_tool_failure.assert_called_once_with(
            flow_type="software_development",
            tool_name="my_tool",
            failure_reason="unknown",
        )

    def test_success_does_not_record_failure_metric(self, tracker):
        tracker.tracker.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_SUCCESS,
            tool_name="gitlab_search",
        )
        tracker.metrics.count_agent_platform_tool_failure.assert_not_called()


class TestHandleTypeErrorResponse:
    def test_returns_error_with_schema(self, tracker):

        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "search_tool"
        mock_tool.args_schema = Mock()
        mock_tool.args_schema.model_json_schema.return_value = {"type": "object"}

        error = TypeError("wrong args")
        result = tracker.tracker.handle_type_error_response(mock_tool, error)

        assert "search_tool" in result
        assert "wrong arguments" in result
        assert "The schema is:" in result

        # Verify failure was tracked
        tracker.client.track_event.assert_called_once()
        call_kwargs = tracker.client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value
        extra = call_kwargs["additional_properties"].extra
        assert extra["error_type"] == "TypeError"
        assert extra["error"] == "wrong args"

    def test_returns_error_without_schema(self, tracker):

        mock_tool = Mock(spec=BaseTool)
        mock_tool.name = "simple_tool"
        mock_tool.args_schema = None

        error = TypeError("wrong args")
        result = tracker.tracker.handle_type_error_response(mock_tool, error)

        assert "does not accept any argument" in result


class TestHandleValidationError:
    def test_returns_validation_error_message(self, tracker):

        error = ValidationError.from_exception_data(
            "ValidationError",
            [{"type": "missing", "loc": ("field",), "input": None}],
        )

        result = tracker.tracker.handle_validation_error("my_tool", error)

        assert "my_tool raised validation error" in result
        tracker.client.track_event.assert_called_once()
        call_kwargs = tracker.client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value
        assert (
            call_kwargs["additional_properties"].extra["error_type"]
            == "ValidationError"
        )


class TestHandleExecutionError:
    def test_returns_runtime_error_message(self, tracker):
        error = Exception("connection timeout")
        result = tracker.tracker.handle_execution_error("my_tool", error)

        assert "runtime exception due to connection timeout" in result
        tracker.client.track_event.assert_called_once()
        call_kwargs = tracker.client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value
        extra = call_kwargs["additional_properties"].extra
        assert extra["error_type"] == "Exception"
        assert extra["error"] == "connection timeout"


class TestHandleToolException:
    def test_returns_tool_exception_message(self, tracker):

        error = ToolException("tool failed")
        result = tracker.tracker.handle_tool_exception("my_tool", error)

        assert "Tool exception occurred due to tool failed" == result
        tracker.client.track_event.assert_called_once()
        call_kwargs = tracker.client.track_event.call_args.kwargs
        assert call_kwargs["event_name"] == EventEnum.WORKFLOW_TOOL_FAILURE.value
        assert (
            call_kwargs["additional_properties"].extra["error_type"] == "ToolException"
        )
