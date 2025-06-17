from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.correlation_id_interceptor import (
    CorrelationIdInterceptor,
    correlation_id,
    gitlab_global_user_id,
)


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
