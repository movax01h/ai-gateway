import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.interceptors.authentication_interceptor import (
    AuthenticationInterceptor,
    current_user,
)


@pytest.fixture
def mock_continuation():
    return AsyncMock()


@pytest.fixture
def handler_call_details():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("authorization", "bearer test-token"),
        ("x-gitlab-authentication-type", "oidc"),
        ("x-gitlab-realm", "test-realm"),
        ("x-gitlab-instance-id", "test-instance-id"),
        ("x-gitlab-global-user-id", "test-global-user-id"),
    )
    return mock_details


@pytest.fixture
def interceptor():
    return AuthenticationInterceptor()


@patch.dict(os.environ, {"DUO_WORKFLOW_AUTH__ENABLED": "false"})
@pytest.mark.asyncio
async def test_intercept_service_auth_disabled(
    interceptor, mock_continuation, handler_call_details
):
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    user = current_user.get()
    assert user.is_authenticated
    assert user.is_debug


@patch.dict(
    os.environ,
    {
        "DUO_WORKFLOW_AUTH__ENABLED": "true",
        "CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service",
    },
)
@patch(
    "duo_workflow_service.interceptors.authentication_interceptor.authenticate",
    return_value=(
        CloudConnectorUser(True, claims=UserClaims(gitlab_realm="test-realm")),
        None,
    ),
)
@pytest.mark.asyncio
async def test_intercept_service_auth_enabled(
    mock_authenticate, interceptor, mock_continuation, handler_call_details
):
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    user = current_user.get()
    assert user.is_authenticated
