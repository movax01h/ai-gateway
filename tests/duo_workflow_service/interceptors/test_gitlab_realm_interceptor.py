from unittest.mock import AsyncMock, MagicMock

import pytest

from duo_workflow_service.interceptors.gitlab_realm_interceptor import (
    GitLabRealmInterceptor,
    gitlab_realm,
)


@pytest.mark.asyncio
async def test_gitlab_realm_interceptor_sets_realm():
    """Test that the interceptor sets the GitLab realm from headers."""
    interceptor = GitLabRealmInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-realm", "saas"),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert gitlab_realm.get() == "saas"

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_gitlab_realm_interceptor_no_realm_header():
    """Test that the interceptor handles missing GitLab realm header."""
    interceptor = GitLabRealmInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    gitlab_realm.set(None)

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert gitlab_realm.get() is None

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"


@pytest.mark.asyncio
async def test_gitlab_realm_interceptor_empty_realm():
    """Test that the interceptor handles empty GitLab realm header."""
    interceptor = GitLabRealmInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-realm", ""),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    gitlab_realm.set(None)

    result = await interceptor.intercept_service(continuation, handler_call_details)

    assert gitlab_realm.get() is None

    continuation.assert_called_once_with(handler_call_details)
    assert result == "mocked_response"
