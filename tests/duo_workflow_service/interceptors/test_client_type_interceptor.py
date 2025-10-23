from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.client_type_interceptor import (
    ClientTypeInterceptor,
    client_type,
)


@pytest.mark.asyncio
async def test_client_type_interceptor_when_header_exists():
    interceptor = ClientTypeInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-client-type", "node-grpc"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert client_type.get() == "node-grpc"

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_client_type_interceptor_when_header_doesnt_exist():
    interceptor = ClientTypeInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert client_type.get() is None

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_client_type_interceptor_when_header_value_empty():
    interceptor = ClientTypeInterceptor()
    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-client-type", ""),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert client_type.get() is None

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"
