from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest

from duo_workflow_service.interceptors.enabled_instance_verbose_ai_logs_interceptor import (
    EnabledInstanceVerboseAiLogsInterceptor,
)
from lib.verbose_ai_logs.context import current_verbose_ai_logs_context


@pytest.fixture(name="reset_context")
def reset_context_fixture():
    """Reset the context variable after each test."""
    token = current_verbose_ai_logs_context.set(False)
    yield
    current_verbose_ai_logs_context.reset(token)


@pytest.fixture(name="mock_handler_call_details")
def mock_handler_call_details_fixture():
    """Create a mock for the handler_call_details."""
    details = MagicMock(spec=grpc.HandlerCallDetails)
    details.invocation_metadata = ()
    return details


@pytest.fixture(name="mock_continuation")
def mock_continuation_fixture():
    """Create a mock for the continuation function."""
    return AsyncMock()


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return EnabledInstanceVerboseAiLogsInterceptor()


class TestEnabledInstanceVerboseAiLogsInterceptor:
    @pytest.mark.asyncio
    async def test_intercept_service_with_enabled_flag(
        self, reset_context, interceptor, mock_handler_call_details, mock_continuation
    ):
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-instance-verbose-ai-logs", "true")
        ]

        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        assert current_verbose_ai_logs_context.get() is True
        mock_continuation.assert_called_once_with(mock_handler_call_details)

    @pytest.mark.asyncio
    async def test_intercept_service_with_disabled_flag(
        self, reset_context, interceptor, mock_handler_call_details, mock_continuation
    ):
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-instance-verbose-ai-logs", "false")
        ]

        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        assert current_verbose_ai_logs_context.get() is False
        mock_continuation.assert_called_once_with(mock_handler_call_details)

    @pytest.mark.asyncio
    async def test_intercept_service_without_flag(
        self, reset_context, interceptor, mock_handler_call_details, mock_continuation
    ):
        mock_handler_call_details.invocation_metadata = []

        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        assert current_verbose_ai_logs_context.get() is False
        mock_continuation.assert_called_once_with(mock_handler_call_details)

    @pytest.mark.asyncio
    async def test_intercept_service_with_other_metadata(
        self, reset_context, interceptor, mock_handler_call_details, mock_continuation
    ):
        mock_handler_call_details.invocation_metadata = [
            ("x-gitlab-enabled-instance-verbose-ai-logs", "true"),
            ("x-gitlab-correlation-id", "test-correlation-id"),
            ("x-gitlab-global-user-id", "123"),
        ]

        await interceptor.intercept_service(
            mock_continuation, mock_handler_call_details
        )

        assert current_verbose_ai_logs_context.get() is True
        mock_continuation.assert_called_once_with(mock_handler_call_details)
