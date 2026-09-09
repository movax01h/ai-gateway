# pylint: disable=pointless-statement
import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import grpc
import pytest
import sentry_sdk
from prometheus_client import CollectorRegistry
from sentry_sdk.transport import Transport
from structlog.testing import capture_logs

from duo_workflow_service.interceptors.monitoring_interceptor import (
    CANCELLED_BEFORE_START,
    GRPCMethodType,
    MonitoringInterceptor,
)
from duo_workflow_service.tracking import MonitoringContext, current_monitoring_context
from lib.language_server import LanguageServerVersion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "service_name",
        "method_name",
        "grpc_type",
        "handler_attr",
        "request_streaming",
        "response_streaming",
    ),
    [
        ("test.Service", "UnaryMethod", "UNARY", "unary_unary", False, False),
        (
            "test.StreamService",
            "StreamUnaryMethod",
            "CLIENT_STREAM",
            "stream_unary",
            True,
            False,
        ),
    ],
)
async def test_interceptor_methods(
    service_name,
    method_name,
    grpc_type,
    handler_attr,
    request_streaming,
    response_streaming,
):
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = f"/{service_name}/{method_name}"
    handler_call_details.invocation_metadata = {"user-agent": "test_agent"}

    mock_handler = Mock()
    setattr(mock_handler, handler_attr, AsyncMock(return_value="response"))
    mock_handler.request_streaming = request_streaming
    mock_handler.response_streaming = response_streaming

    continuation.return_value = mock_handler
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.OK

    result = await interceptor.intercept_service(continuation, handler_call_details)
    assert result is not None

    handler_func = getattr(result, handler_attr)

    with capture_logs() as cap_logs:
        response = await handler_func(None, mock_context)

    assert response == "response"

    total_calls = registry.get_sample_value(
        "grpc_server_handled_total",
        {
            "grpc_type": grpc_type,
            "grpc_service": service_name,
            "grpc_method": method_name,
            "grpc_code": "OK",
            "gitlab_version": "unknown",
            "client_type": "unknown",
            "lsp_version": "unknown",
            "flow_type": "unknown",
            "gitlab_realm": "unknown",
            "is_gitlab_team_member": "unknown",
        },
    )

    assert total_calls == 1.0

    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == f"Finished {method_name} RPC"
    assert cap_logs[0]["grpc_service_name"] == service_name
    assert cap_logs[0]["grpc_method_name"] == method_name
    assert cap_logs[0]["user_agent"] == "test_agent"


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.interceptors.monitoring_interceptor.MonitoringContext",
    return_value=MonitoringContext(
        workflow_last_gitlab_status="running",
        workflow_stop_reason="stopped by client",
        workflow_definition="chat",
    ),
)
@patch(
    "duo_workflow_service.interceptors.monitoring_interceptor.language_server_version",
)
@patch(
    "duo_workflow_service.interceptors.monitoring_interceptor.client_type",
)
@patch(
    "duo_workflow_service.interceptors.monitoring_interceptor.current_feature_flag_context",
)
@pytest.mark.parametrize(
    (
        "service_name",
        "method_name",
        "grpc_type",
        "handler_attr",
        "request_streaming",
        "response_streaming",
    ),
    [
        (
            "test.Service",
            "UnaryStreamMethod",
            "SERVER_STREAM",
            "unary_stream",
            False,
            True,
        ),
        (
            "test.StreamService",
            "StreamStreamMethod",
            "BIDI_STREAM",
            "stream_stream",
            True,
            True,
        ),
    ],
)
async def test_streaming_interceptor_methods(
    mock_feature_flag_context,
    mock_client_type,
    mock_language_server_version,
    _mock_monitoring_context,
    service_name,
    method_name,
    grpc_type,
    handler_attr,
    request_streaming,
    response_streaming,
):
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = f"/{service_name}/{method_name}"
    handler_call_details.invocation_metadata = {}

    mock_language_server_version.get.return_value = LanguageServerVersion("0.0.1")
    mock_client_type.get.return_value = "node-grpc"
    mock_feature_flag_context.get.return_value = {"ai_context_compaction"}

    async def _stream_generator(_req, _ctx):
        yield "Stream"
        yield "content"

    mock_handler = Mock()
    streamed_response = MagicMock(side_effect=_stream_generator)

    setattr(mock_handler, handler_attr, streamed_response)
    mock_handler.request_streaming = request_streaming
    mock_handler.response_streaming = response_streaming

    continuation.return_value = mock_handler
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.OK

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert result is not None

    handler_func = getattr(result, handler_attr)

    content = []
    with capture_logs() as cap_logs:
        async for chunk in handler_func(None, mock_context):
            content.append(chunk)
    assert content == ["Stream", "content"]

    total_calls = registry.get_sample_value(
        "grpc_server_handled_total",
        {
            "grpc_type": grpc_type,
            "grpc_service": service_name,
            "grpc_method": method_name,
            "grpc_code": "OK",
            "gitlab_version": "unknown",
            "client_type": "unknown",
            "lsp_version": "unknown",
            "gitlab_realm": "unknown",
            "flow_type": "chat",
            "is_gitlab_team_member": "unknown",
        },
    )

    assert total_calls == 1.0

    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == f"Finished {method_name} RPC"
    assert cap_logs[0]["grpc_service_name"] == service_name
    assert cap_logs[0]["grpc_method_name"] == method_name
    assert cap_logs[0]["workflow_last_gitlab_status"] == "running"
    assert cap_logs[0]["workflow_stop_reason"] == "stopped by client"
    assert cap_logs[0]["language_server_version"] == "0.0.1"
    assert cap_logs[0]["gitlab_client_type"] == "node-grpc"
    assert cap_logs[0]["feature_flags"] == {"ai_context_compaction"}


@pytest.mark.asyncio
async def test_interceptor_handles_exception():
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = "/test.Service/ErrorMethod"
    handler_call_details.invocation_metadata = {}

    mock_handler = Mock()
    mock_handler.unary_unary = AsyncMock(side_effect=Exception("Test Exception"))
    mock_handler.request_streaming = False
    mock_handler.response_streaming = False

    continuation.return_value = mock_handler
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.OK

    with pytest.raises(Exception, match="Test Exception"), capture_logs() as cap_logs:
        result = await interceptor.intercept_service(continuation, handler_call_details)
        assert result is not None

        await result.unary_unary(None, mock_context)

    total_calls = registry.get_sample_value(
        "grpc_server_handled_total",
        {
            "grpc_type": "UNARY",
            "grpc_service": "test.Service",
            "grpc_method": "ErrorMethod",
            "grpc_code": "UNKNOWN",
            "gitlab_version": "unknown",
            "client_type": "unknown",
            "lsp_version": "unknown",
            "gitlab_realm": "unknown",
            "flow_type": "unknown",
            "is_gitlab_team_member": "unknown",
        },
    )

    assert total_calls == 1.0
    assert len(cap_logs) == 2
    assert cap_logs[0]["event"] == "Test Exception"
    assert cap_logs[0]["exception_class"] == "Exception"
    assert cap_logs[1]["event"] == "Finished ErrorMethod RPC"


@pytest.mark.asyncio
async def test_interceptor_stream_handles_exception():
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = "/test.Service/StreamErrorMethod"
    handler_call_details.invocation_metadata = {}

    mock_handler = Mock()
    mock_handler.stream_stream = MagicMock(side_effect=BaseException("Test Exception"))
    mock_handler.request_streaming = True
    mock_handler.response_streaming = True

    continuation.return_value = mock_handler
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.OK

    with (
        pytest.raises(BaseException, match="Test Exception"),
        capture_logs() as cap_logs,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)
        assert result is not None

        async for _ in result.stream_stream(None, mock_context):
            _

    total_calls = registry.get_sample_value(
        "grpc_server_handled_total",
        {
            "grpc_type": "BIDI_STREAM",
            "grpc_service": "test.Service",
            "grpc_method": "StreamErrorMethod",
            "grpc_code": "UNKNOWN",
            "gitlab_version": "unknown",
            "client_type": "unknown",
            "lsp_version": "unknown",
            "gitlab_realm": "unknown",
            "flow_type": "unknown",
            "is_gitlab_team_member": "unknown",
        },
    )

    assert total_calls == 1.0
    assert len(cap_logs) == 2
    assert cap_logs[0]["event"] == "Test Exception"
    assert cap_logs[0]["exception_class"] == "BaseException"
    assert cap_logs[1]["event"] == "Finished StreamErrorMethod RPC"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "no_start_reason",
    ["NO_START_REQUEST", "EMPTY_START_REQUEST"],
)
async def test_interceptor_logs_info_and_skips_finished_log_when_workflow_never_started(
    no_start_reason,
):
    """When workflow_no_start_reason is set on MonitoringContext the interceptor must:
    - emit a single info-level 'connection closed before workflow started' log with the reason
    - not emit 'Finished RPC'
    - not increment the Prometheus counter
    """
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = "/test.Service/ExecuteWorkflow"
    handler_call_details.invocation_metadata = {"user-agent": "test_agent"}

    async def _never_started_stream_handler(_req, _ctx):
        # Simulate the usage_quota wrapper setting the reason and returning without
        # yielding anything.
        current_monitoring_context.get().workflow_no_start_reason = no_start_reason
        return
        yield  # make it an async generator

    mock_handler = Mock()
    mock_handler.stream_stream = MagicMock(side_effect=_never_started_stream_handler)
    mock_handler.request_streaming = True
    mock_handler.response_streaming = True

    continuation.return_value = mock_handler
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.OK

    result = await interceptor.intercept_service(continuation, handler_call_details)
    assert result is not None

    with capture_logs() as cap_logs:
        async for _ in result.stream_stream(None, mock_context):
            pass

    assert len(cap_logs) == 1
    assert cap_logs[0]["log_level"] == "info"
    assert cap_logs[0]["event"] == "connection closed before workflow started"
    assert cap_logs[0]["no_start_reason"] == no_start_reason
    assert cap_logs[0]["user_agent"] == "test_agent"

    total_calls = registry.get_sample_value(
        "grpc_server_handled_total",
        {
            "grpc_type": "BIDI_STREAM",
            "grpc_service": "test.Service",
            "grpc_method": "ExecuteWorkflow",
            "grpc_code": "OK",
            "gitlab_version": "unknown",
            "client_type": "unknown",
            "lsp_version": "unknown",
            "gitlab_realm": "unknown",
            "flow_type": "unknown",
            "is_gitlab_team_member": "unknown",
        },
    )
    assert total_calls is None, (
        "Expected no Prometheus counter increment when workflow never started"
    )


class _CaptureTransport(Transport):
    def __init__(self):
        super().__init__()
        self.events = []

    def capture_envelope(self, envelope):
        for item in envelope.items:
            if payload := item.payload.json:
                self.events.append(payload)

    def flush(self, *args, **kwargs):
        pass

    def kill(self):
        pass


@pytest.fixture(name="sentry_events")
def sentry_events_fixture():
    # Restoring the previous client rather than re-initialising: `sentry_sdk.init`
    # installs the default integrations process-wide and they are never unpatched,
    # which would leak into every later test in this xdist worker.
    old_client = sentry_sdk.get_global_scope().client
    transport = _CaptureTransport()
    sentry_sdk.init(
        dsn="https://public@example.ingest.sentry.io/1",
        traces_sample_rate=1.0,
        default_integrations=False,
        transport=transport,
    )
    current_monitoring_context.set(MonitoringContext())

    try:
        yield transport.events
    finally:
        sentry_sdk.get_global_scope().set_client(old_client)


def _root_span(grpc_type, grpc_method_name="ExecuteWorkflow"):
    return MonitoringInterceptor(registry=CollectorRegistry()).sentry_root_span(
        grpc_type=grpc_type,
        grpc_service_name="DuoWorkflow",
        grpc_method_name=grpc_method_name,
        invocation_metadata={},
    )


def test_sentry_root_span_skips_unary_rpcs(sentry_events):
    with _root_span(GRPCMethodType.UNARY):
        pass

    sentry_sdk.flush()

    assert sentry_events == []


def test_sentry_root_span_skips_streaming_rpcs_when_sentry_is_not_initialized():
    body_ran = False

    with (
        patch("sentry_sdk.is_initialized", return_value=False),
        patch("sentry_sdk.start_transaction") as start_transaction,
    ):
        with _root_span(GRPCMethodType.BIDI_STREAMING):
            body_ran = True

    assert body_ran
    start_transaction.assert_not_called()


def test_sentry_root_span_opens_transaction_for_streaming_rpcs(sentry_events):
    with _root_span(GRPCMethodType.BIDI_STREAMING):
        pass

    sentry_sdk.flush()

    assert [e["transaction"] for e in sentry_events] == ["/DuoWorkflow/ExecuteWorkflow"]


def test_sentry_root_span_tags_bounded_fields_and_stores_ids_as_data(sentry_events):
    with _root_span(GRPCMethodType.BIDI_STREAMING):
        context = current_monitoring_context.get()
        context.workflow_definition = "software_development"
        context.workflow_id = "run-1"
        context.flow_id = "catalog/42"

    sentry_sdk.flush()
    event = sentry_events[0]

    assert event["tags"]["flow_type"] == "software_development"
    assert "workflow_id" not in event["tags"]
    assert event["contexts"]["trace"]["data"]["workflow_id"] == "run-1"
    assert event["contexts"]["trace"]["data"]["flow_id"] == "catalog/42"


def test_sentry_root_span_tags_when_the_rpc_raises(sentry_events):
    with pytest.raises(RuntimeError):
        with _root_span(GRPCMethodType.BIDI_STREAMING):
            current_monitoring_context.get().workflow_stop_reason = "cancelled"
            raise RuntimeError("boom")

    sentry_sdk.flush()

    assert sentry_events[0]["tags"]["workflow_stop_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_sentry_root_span_keeps_concurrent_rpcs_isolated(sentry_events):
    # Mirrors the server: the task that accepts the RPCs already has a current scope,
    # so every RPC task inherits the same scope object unless the span forks it.
    sentry_sdk.get_current_scope()

    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    first_child_done = asyncio.Event()

    async def first_rpc():
        with _root_span(GRPCMethodType.BIDI_STREAMING, "FirstFlow"):
            first_entered.set()
            await second_entered.wait()

            with sentry_sdk.start_span(op="gen_ai.chat", name="child-of-first"):
                pass

            first_child_done.set()

    async def second_rpc():
        await first_entered.wait()

        with _root_span(GRPCMethodType.BIDI_STREAMING, "SecondFlow"):
            second_entered.set()
            await first_child_done.wait()

            with sentry_sdk.start_span(op="gen_ai.chat", name="child-of-second"):
                pass

    await asyncio.gather(first_rpc(), second_rpc())
    sentry_sdk.flush()

    spans_by_transaction = {
        event["transaction"]: [span["description"] for span in event["spans"]]
        for event in sentry_events
    }

    assert spans_by_transaction == {
        "/DuoWorkflow/FirstFlow": ["child-of-first"],
        "/DuoWorkflow/SecondFlow": ["child-of-second"],
    }


@pytest.mark.asyncio
async def test_exceptions_are_logged_while_the_transaction_is_open(sentry_events):
    interceptor = MonitoringInterceptor(registry=CollectorRegistry())
    handler_call_details = Mock()
    handler_call_details.method = "/DuoWorkflow/ExecuteWorkflow"
    handler_call_details.invocation_metadata = {}

    async def failing_behavior(_request_iterator, _servicer_context):
        raise RuntimeError("boom")
        yield  # pylint: disable=unreachable

    upstream_handler = Mock()
    upstream_handler.stream_stream = failing_behavior
    upstream_handler.request_streaming = True
    upstream_handler.response_streaming = True

    continuation = AsyncMock(return_value=upstream_handler)
    servicer_context = Mock()
    servicer_context.code.return_value = grpc.StatusCode.UNKNOWN

    handler = await interceptor.intercept_service(continuation, handler_call_details)

    spans_when_logged = []

    def record_span(_exception, *_args, **_kwargs):
        spans_when_logged.append(sentry_sdk.get_current_scope().span)

    with patch(
        "duo_workflow_service.interceptors.monitoring_interceptor.log_exception",
        side_effect=record_span,
    ):
        with pytest.raises(RuntimeError):
            async for _response in handler.stream_stream(None, servicer_context):
                pass

    assert len(spans_when_logged) == 1
    # An error captured after the transaction closes loses its link to it.
    assert getattr(spans_when_logged[0], "name", None) == "/DuoWorkflow/ExecuteWorkflow"


def _execute_workflow_stream_handler(handler_fn, method_name="ExecuteWorkflow"):
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = f"/test.Service/{method_name}"
    handler_call_details.invocation_metadata = {}

    mock_handler = Mock()
    mock_handler.stream_stream = MagicMock(side_effect=handler_fn)
    mock_handler.request_streaming = True
    mock_handler.response_streaming = True
    continuation.return_value = mock_handler
    return continuation, handler_call_details


def _handled_total(registry, grpc_code, flow_type, method_name="ExecuteWorkflow"):
    return registry.get_sample_value(
        "grpc_server_handled_total",
        {
            "grpc_type": "BIDI_STREAM",
            "grpc_service": "test.Service",
            "grpc_method": method_name,
            "grpc_code": grpc_code,
            "gitlab_version": "unknown",
            "client_type": "unknown",
            "lsp_version": "unknown",
            "gitlab_realm": "unknown",
            "flow_type": flow_type,
            "is_gitlab_team_member": "unknown",
        },
    )


@pytest.mark.asyncio
async def test_interceptor_skips_counter_when_stream_cancelled_before_workflow_started():
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)

    async def _cancelled_before_start(_req, _ctx):
        raise asyncio.CancelledError()
        yield  # make it an async generator

    continuation, handler_call_details = _execute_workflow_stream_handler(
        _cancelled_before_start
    )
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.CANCELLED

    result = await interceptor.intercept_service(continuation, handler_call_details)

    with pytest.raises(asyncio.CancelledError), capture_logs() as cap_logs:
        async for _ in result.stream_stream(None, mock_context):
            pass

    assert len(cap_logs) == 1
    assert cap_logs[0]["log_level"] == "info"
    assert cap_logs[0]["event"] == "connection closed before workflow started"
    assert cap_logs[0]["no_start_reason"] == CANCELLED_BEFORE_START
    assert _handled_total(registry, "CANCELLED", "unknown") is None


@pytest.mark.asyncio
async def test_interceptor_counts_cancellation_after_workflow_started():
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)

    async def _cancelled_mid_run(_req, _ctx):
        context = current_monitoring_context.get()
        context.workflow_id = "wf-1"
        context.workflow_definition = "chat"
        raise asyncio.CancelledError()
        yield  # make it an async generator

    continuation, handler_call_details = _execute_workflow_stream_handler(
        _cancelled_mid_run
    )
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.CANCELLED

    result = await interceptor.intercept_service(continuation, handler_call_details)

    with pytest.raises(asyncio.CancelledError), capture_logs() as cap_logs:
        async for _ in result.stream_stream(None, mock_context):
            pass

    assert cap_logs[-1]["event"] == "Finished ExecuteWorkflow RPC"
    assert _handled_total(registry, "CANCELLED", "chat") == 1.0


@pytest.mark.asyncio
async def test_interceptor_counts_cancellation_of_other_bidi_streams_without_workflow_id():
    """TrackSelfHostedExecuteWorkflow never sets workflow_id, so its cancellations must still be counted."""
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)

    async def _cancelled_tracking_stream(_req, _ctx):
        raise asyncio.CancelledError()
        yield  # make it an async generator

    continuation, handler_call_details = _execute_workflow_stream_handler(
        _cancelled_tracking_stream, method_name="TrackSelfHostedExecuteWorkflow"
    )
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.CANCELLED

    result = await interceptor.intercept_service(continuation, handler_call_details)

    with pytest.raises(asyncio.CancelledError), capture_logs() as cap_logs:
        async for _ in result.stream_stream(None, mock_context):
            pass

    assert cap_logs[-1]["event"] == "Finished TrackSelfHostedExecuteWorkflow RPC"
    assert (
        _handled_total(
            registry, "CANCELLED", "unknown", "TrackSelfHostedExecuteWorkflow"
        )
        == 1.0
    )
