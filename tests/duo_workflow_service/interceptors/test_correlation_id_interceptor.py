from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import WebSocket

from duo_workflow_service.interceptors.correlation_id_interceptor import (
    CorrelationIdInterceptor,
    CorrelationIdMiddleware,
    correlation_id,
    gitlab_global_user_id,
)

# --------- gRPC Interceptor Tests ---------


@pytest.fixture
def mock_continuation():
    return AsyncMock()


@pytest.fixture
def interceptor():
    return CorrelationIdInterceptor()


@pytest.fixture
def handler_call_details():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-request-id", "test-request"),
        ("x-gitlab-global-user-id", "test-user"),
    )
    return mock_details


@pytest.mark.asyncio
async def test_log_interceptor_with_existing_request_id(
    interceptor, mock_continuation, handler_call_details
):
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    current_correlation_id = correlation_id.get()
    assert current_correlation_id == "test-request"


@pytest.mark.asyncio
async def test_log_interceptor_with_existing_gitlab_global_user_id(
    interceptor, mock_continuation, handler_call_details
):
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    current_gitlab_global_user_id = gitlab_global_user_id.get()
    assert current_gitlab_global_user_id == "test-user"


@pytest.mark.asyncio
async def test_log_interceptor(interceptor, mock_continuation, handler_call_details):
    handler_call_details.invocation_metadata = ()
    with mock.patch("uuid.uuid4", return_value="test-uuid"):
        await interceptor.intercept_service(mock_continuation, handler_call_details)
        current_correlation_id = correlation_id.get()
        assert current_correlation_id == "test-uuid"


# --------- WebSocket Middleware Tests ---------


@pytest.fixture
def middleware():
    return CorrelationIdMiddleware()


@pytest.fixture
def mock_websocket():
    def _create_websocket(headers=None):
        mock_ws = MagicMock(spec=WebSocket)
        mock_ws.headers = headers or {}
        return mock_ws

    return _create_websocket


class TestCorrelationIdMiddleware:
    """Tests for WebSocket CorrelationIdMiddleware."""

    @pytest.mark.parametrize(
        "test_case, headers, expected_correlation_id, expected_user_id",
        [
            # Test with existing request headers
            (
                "existing_headers",
                {
                    "x-request-id": "test-request",
                    "x-gitlab-global-user-id": "test-user",
                },
                "test-request",
                "test-user",
            ),
            # Test with no request headers (should generate UUID)
            (
                "no_headers",
                {},
                "test-uuid",  # From mocked uuid4
                "undefined",
            ),
            # Test with only correlation ID
            (
                "only_correlation_id",
                {"x-request-id": "only-correlation"},
                "only-correlation",
                "undefined",
            ),
            # Test with only user ID
            (
                "only_user_id",
                {"x-gitlab-global-user-id": "only-user"},
                "test-uuid",  # Should generate ID
                "only-user",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_middleware(
        self,
        middleware,
        mock_websocket,
        test_case,
        headers,
        expected_correlation_id,
        expected_user_id,
    ):
        # Create mock WebSocket with headers
        websocket = mock_websocket(headers)

        # Run the test with UUID patch
        with mock.patch("uuid.uuid4", return_value="test-uuid"):
            # Call middleware
            await middleware(websocket)

            # Verify context variables are set correctly
            assert correlation_id.get() == expected_correlation_id
            assert gitlab_global_user_id.get() == expected_user_id
