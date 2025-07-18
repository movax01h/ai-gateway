from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.gitlab_version_interceptor import (
    GitLabVersionInterceptor,
    gitlab_version,
)


@pytest.mark.asyncio
async def test_gitlab_version_interceptor_sets_version():
    """Test that the interceptor sets the GitLab version from headers."""
    interceptor = GitLabVersionInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-version", "17.5.2"),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert gitlab_version.get() == "17.5.2"

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_gitlab_version_interceptor_no_version_header():
    """Test that the interceptor handles missing GitLab version header."""
    interceptor = GitLabVersionInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    gitlab_version.set(None)

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert gitlab_version.get() is None

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_gitlab_version_interceptor_empty_version():
    """Test that the interceptor handles empty GitLab version header."""
    interceptor = GitLabVersionInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-version", ""),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    gitlab_version.set(None)

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert gitlab_version.get() is None

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"
