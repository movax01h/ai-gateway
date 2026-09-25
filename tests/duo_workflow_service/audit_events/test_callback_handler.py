import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from duo_workflow_service.audit_events.callback_handler import (
    AuditEventCallbackHandler,
)
from duo_workflow_service.audit_events.collector import AuditEventCollector
from duo_workflow_service.audit_events.event_types import (
    LlmInputSentEvent,
    LlmRequestFailedEvent,
    LlmResponseReceivedEvent,
    ToolExecutionFailedEvent,
    ToolInvokedEvent,
    ToolResponseReceivedEvent,
)
from lib.context.approval_sources import (
    approval_sources,
    init_approval_sources,
    record_approval_policy_ref,
    record_approval_source,
)


@pytest.fixture(name="collector")
def collector_fixture():
    collector = MagicMock(spec=AuditEventCollector)
    return collector


@pytest.fixture(name="handler")
def handler_fixture(collector):
    return AuditEventCallbackHandler(collector=collector, workflow_id="wf-1")


class TestOnChatModelStart:
    @pytest.mark.asyncio
    async def test_captures_llm_input_event(self, handler, collector):
        run_id = uuid4()
        messages = [[AIMessage(content="hello")]]
        await handler.on_chat_model_start(
            serialized={"kwargs": {"model": "claude-3"}},
            messages=messages,
            run_id=run_id,
        )
        collector.capture.assert_called_once()
        event = collector.capture.call_args[0][0]
        assert isinstance(event, LlmInputSentEvent)
        assert event.model_name == "claude-3"
        assert event.workflow_id == "wf-1"

    @pytest.mark.asyncio
    async def test_extracts_model_from_invocation_params(self, handler, collector):
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[AIMessage(content="hello")]],
            run_id=run_id,
            invocation_params={"model": "claude-3-opus"},
        )
        event = collector.capture.call_args[0][0]
        assert event.model_name == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_extracts_model_from_invocation_params_model_name(
        self, handler, collector
    ):
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[AIMessage(content="hello")]],
            run_id=run_id,
            invocation_params={"model_name": "gpt-4"},
        )
        event = collector.capture.call_args[0][0]
        assert event.model_name == "gpt-4"

    @pytest.mark.asyncio
    async def test_extracts_model_from_serialized_kwargs(self, handler, collector):
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={"kwargs": {"model": "claude-3-sonnet"}},
            messages=[[AIMessage(content="hello")]],
            run_id=run_id,
        )
        event = collector.capture.call_args[0][0]
        assert event.model_name == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_model_name_fallback_to_unknown(self, handler, collector):
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[AIMessage(content="hello")]],
            run_id=run_id,
        )
        event = collector.capture.call_args[0][0]
        assert event.model_name == "unknown"

    @pytest.mark.asyncio
    async def test_tracks_start_time(self, handler):
        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={},
            messages=[[AIMessage(content="test")]],
            run_id=run_id,
        )
        assert str(run_id) in handler._llm_start_times


class TestOnLlmEnd:
    @pytest.mark.asyncio
    async def test_captures_response_event(self, handler, collector):
        run_id = uuid4()
        handler._llm_start_times[str(run_id)] = time.monotonic()
        handler._llm_model_names[str(run_id)] = "claude-3"
        message = AIMessage(
            content="response text",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            response_metadata={"stop_reason": "end_turn"},
        )
        generation = ChatGeneration(message=message, text="response text")
        generation.generation_info = {"finish_reason": "end_turn"}
        response = LLMResult(generations=[[generation]])
        await handler.on_llm_end(response=response, run_id=run_id)
        collector.capture.assert_called_once()
        event = collector.capture.call_args[0][0]
        assert isinstance(event, LlmResponseReceivedEvent)
        assert event.model_name == "claude-3"
        assert event.response_content == "response text"
        assert event.prompt_token_count == 10
        assert event.completion_token_count == 20
        assert event.finish_reason == "end_turn"
        assert event.latency_ms is not None

    @pytest.mark.asyncio
    async def test_handles_missing_start_time(self, handler, collector):
        run_id = uuid4()
        handler._llm_model_names[str(run_id)] = "claude-3"
        response = LLMResult(generations=[[]])
        await handler.on_llm_end(response=response, run_id=run_id)
        event = collector.capture.call_args[0][0]
        assert event.latency_ms is None

    @pytest.mark.asyncio
    async def test_cleans_up_tracking_state(self, handler):
        run_id = uuid4()
        handler._llm_start_times[str(run_id)] = 0.0
        handler._llm_model_names[str(run_id)] = "claude-3"
        response = LLMResult(generations=[[]])
        await handler.on_llm_end(response=response, run_id=run_id)
        assert str(run_id) not in handler._llm_start_times
        assert str(run_id) not in handler._llm_model_names


class TestOnToolStart:
    @pytest.mark.asyncio
    async def test_captures_tool_invoked_event(self, handler, collector):
        run_id = uuid4()
        await handler.on_tool_start(
            serialized={"name": "read_file"},
            input_str="test input",
            run_id=run_id,
            inputs={"path": "/src/main.py"},
        )
        collector.capture.assert_called_once()
        event = collector.capture.call_args[0][0]
        assert isinstance(event, ToolInvokedEvent)
        assert event.tool_name == "read_file"
        assert event.tool_args == {"path": "/src/main.py"}

    @pytest.mark.asyncio
    async def test_tracks_tool_name(self, handler):
        run_id = uuid4()
        await handler.on_tool_start(
            serialized={"name": "write_file"},
            input_str="",
            run_id=run_id,
        )
        assert handler._tool_names[str(run_id)] == "write_file"

    @pytest.mark.asyncio
    async def test_captures_recorded_approval_source(self, handler, collector):
        # Silent-reuse crux: a prior approval decision recorded the source under
        # the tool call id; on_tool_start must attach it to the emitted event
        # even though the callback has no other view of the approval.
        init_approval_sources()
        record_approval_source("call-42", "session_approval")
        record_approval_policy_ref("call-42", {"origin": "file"})
        try:
            await handler.on_tool_start(
                serialized={"name": "read_file"},
                input_str="",
                run_id=uuid4(),
                inputs={"path": "/a"},
                tool_call_id="call-42",
            )
        finally:
            approval_sources.set(None)

        event = collector.capture.call_args[0][0]
        assert isinstance(event, ToolInvokedEvent)
        assert event.approval_source == "session_approval"
        assert event.policy_ref == {"origin": "file"}

    @pytest.mark.asyncio
    async def test_approval_source_none_when_unrecorded(self, handler, collector):
        init_approval_sources()
        try:
            await handler.on_tool_start(
                serialized={"name": "read_file"},
                input_str="",
                run_id=uuid4(),
                inputs={"path": "/a"},
                tool_call_id="unknown-call",
            )
        finally:
            approval_sources.set(None)

        event = collector.capture.call_args[0][0]
        assert event.approval_source is None
        assert event.policy_ref is None


class TestOnToolEnd:
    @pytest.mark.asyncio
    async def test_captures_tool_response_event(self, handler, collector):
        run_id = uuid4()
        handler._tool_names[str(run_id)] = "read_file"
        await handler.on_tool_end(output="file contents here", run_id=run_id)
        collector.capture.assert_called_once()
        event = collector.capture.call_args[0][0]
        assert isinstance(event, ToolResponseReceivedEvent)
        assert event.tool_name == "read_file"
        assert event.response_content == "file contents here"
        assert event.response_length == len("file contents here")

    @pytest.mark.asyncio
    async def test_cleans_up_tool_name(self, handler):
        run_id = uuid4()
        handler._tool_names[str(run_id)] = "read_file"
        await handler.on_tool_end(output="output", run_id=run_id)
        assert str(run_id) not in handler._tool_names


class TestOnToolError:
    @pytest.mark.asyncio
    async def test_captures_tool_failed_event(self, handler, collector):
        run_id = uuid4()
        error = ValueError("file not found")
        await handler.on_tool_error(error=error, run_id=run_id, name="read_file")
        collector.capture.assert_called_once()
        event = collector.capture.call_args[0][0]
        assert isinstance(event, ToolExecutionFailedEvent)
        assert event.error_type == "ValueError"
        assert event.error_message == "file not found"

    @pytest.mark.asyncio
    async def test_uses_tracked_tool_name(self, handler, collector):
        run_id = uuid4()
        handler._tool_names[str(run_id)] = "write_file"
        await handler.on_tool_error(error=RuntimeError("boom"), run_id=run_id)
        event = collector.capture.call_args[0][0]
        assert event.tool_name == "write_file"
        assert str(run_id) not in handler._tool_names

    @pytest.mark.asyncio
    async def test_falls_back_to_kwargs_name(self, handler, collector):
        run_id = uuid4()
        await handler.on_tool_error(
            error=RuntimeError("boom"), run_id=run_id, name="fallback_tool"
        )
        event = collector.capture.call_args[0][0]
        assert event.tool_name == "fallback_tool"


class TestOnLlmError:
    @pytest.mark.asyncio
    async def test_captures_llm_request_failed_event(self, handler, collector):
        run_id = uuid4()
        handler._llm_model_names[str(run_id)] = "claude-3"
        error = ConnectionError("upstream timeout")
        await handler.on_llm_error(error=error, run_id=run_id)
        collector.capture.assert_called_once()
        event = collector.capture.call_args[0][0]
        assert isinstance(event, LlmRequestFailedEvent)
        assert event.model_name == "claude-3"
        assert event.error_type == "ConnectionError"
        assert event.error_message == "upstream timeout"
        assert event.workflow_id == "wf-1"

    @pytest.mark.asyncio
    async def test_computes_latency_when_start_time_tracked(self, handler, collector):
        run_id = uuid4()
        handler._llm_start_times[str(run_id)] = time.monotonic()
        handler._llm_model_names[str(run_id)] = "claude-3"
        await handler.on_llm_error(error=RuntimeError("fail"), run_id=run_id)
        event = collector.capture.call_args[0][0]
        assert event.latency_ms is not None
        assert event.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_latency_is_none_when_no_start_time(self, handler, collector):
        run_id = uuid4()
        handler._llm_model_names[str(run_id)] = "claude-3"
        await handler.on_llm_error(error=RuntimeError("fail"), run_id=run_id)
        event = collector.capture.call_args[0][0]
        assert event.latency_ms is None

    @pytest.mark.asyncio
    async def test_cleans_up_tracking_state(self, handler):
        run_id = uuid4()
        handler._llm_start_times[str(run_id)] = time.monotonic()
        handler._llm_model_names[str(run_id)] = "claude-3"
        await handler.on_llm_error(error=RuntimeError("fail"), run_id=run_id)
        assert str(run_id) not in handler._llm_start_times
        assert str(run_id) not in handler._llm_model_names

    @pytest.mark.asyncio
    async def test_model_name_falls_back_to_unknown(self, handler, collector):
        run_id = uuid4()
        await handler.on_llm_error(error=RuntimeError("fail"), run_id=run_id)
        event = collector.capture.call_args[0][0]
        assert event.model_name == "unknown"
