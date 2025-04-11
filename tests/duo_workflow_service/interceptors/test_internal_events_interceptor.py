from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.internal_events_interceptor import (
    InternalEventsInterceptor,
)
from duo_workflow_service.internal_events import current_event_context


@pytest.fixture
def mock_continuation():
    return AsyncMock()


@pytest.fixture
def interceptor():
    return InternalEventsInterceptor()


@pytest.fixture
def handler_call_details():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("x-gitlab-realm", "test-realm"),
        ("x-gitlab-instance-id", "test-instance-id"),
        ("x-gitlab-global-user-id", "test-global-user-id"),
        ("x-gitlab-host-name", "test-gitlab-host"),
    )
    return mock_details


@pytest.mark.asyncio
async def test_interceptor_with_internal_events_disabled(
    interceptor, mock_continuation, handler_call_details
):
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    event_context = current_event_context.get()
    assert event_context.realm == "test-realm"
    assert event_context.instance_id == "test-instance-id"
    assert event_context.global_user_id == "test-global-user-id"
    assert event_context.host_name == "test-gitlab-host"
