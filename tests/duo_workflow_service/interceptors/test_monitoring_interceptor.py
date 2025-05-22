from unittest.mock import AsyncMock, MagicMock, Mock

import grpc
import pytest
from prometheus_client import CollectorRegistry
from structlog.testing import capture_logs

from duo_workflow_service.interceptors.monitoring_interceptor import (
    MonitoringInterceptor,
)


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
        },
    )

    assert total_calls == 1.0

    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == f"Finished {method_name} RPC"
    assert cap_logs[0]["grpc_service_name"] == service_name
    assert cap_logs[0]["grpc_method_name"] == method_name
    assert cap_logs[0]["user_agent"] == "test_agent"


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
        },
    )
    assert total_calls == 1.0

    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == f"Finished {method_name} RPC"
    assert cap_logs[0]["grpc_service_name"] == service_name
    assert cap_logs[0]["grpc_method_name"] == method_name


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
        },
    )

    assert total_calls == 1.0
    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == f"Finished ErrorMethod RPC"
    assert cap_logs[0]["exception_message"] == "Test Exception"
    assert cap_logs[0]["exception_class"] == "Exception"


@pytest.mark.asyncio
async def test_interceptor_stream_handles_exception():
    registry = CollectorRegistry()
    interceptor = MonitoringInterceptor(registry=registry)
    continuation = AsyncMock()
    handler_call_details = Mock()
    handler_call_details.method = "/test.Service/StreamErrorMethod"
    handler_call_details.invocation_metadata = {}

    mock_handler = Mock()
    mock_handler.stream_stream = MagicMock(side_effect=Exception("Test Exception"))
    mock_handler.request_streaming = True
    mock_handler.response_streaming = True

    continuation.return_value = mock_handler
    mock_context = Mock()
    mock_context.code.return_value = grpc.StatusCode.OK

    with pytest.raises(Exception, match="Test Exception"), capture_logs() as cap_logs:
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
        },
    )

    assert total_calls == 1.0
    assert len(cap_logs) == 1
    assert cap_logs[0]["event"] == f"Finished StreamErrorMethod RPC"
    assert cap_logs[0]["exception_message"] == "Test Exception"
    assert cap_logs[0]["exception_class"] == "Exception"
