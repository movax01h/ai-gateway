import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.interceptors.authentication_interceptor import (
    AuthenticationInterceptor,
    current_user,
)


@pytest.fixture(name="mock_continuation")
def mock_continuation_fixture():
    return AsyncMock()


@pytest.fixture(name="handler_call_details")
def handler_call_details_fixture():
    mock_details = MagicMock()
    mock_details.invocation_metadata = (
        ("authorization", "bearer test-token"),
        ("x-gitlab-authentication-type", "oidc"),
        ("x-gitlab-realm", "test-realm"),
        ("x-gitlab-instance-id", "test-instance-id"),
        ("x-gitlab-global-user-id", "test-global-user-id"),
    )
    return mock_details


@pytest.fixture(name="interceptor")
def interceptor_fixture():
    return AuthenticationInterceptor()


@pytest.mark.parametrize(
    "method",
    ["/grpc.health.v1.Health/Check", "/grpc.health.v1.Health/Watch"],
)
@pytest.mark.asyncio
async def test_intercept_service_health_check_skips_auth(
    method, interceptor, mock_continuation, handler_call_details
):
    handler_call_details.method = method
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    mock_continuation.assert_awaited_once_with(handler_call_details)


@pytest.mark.parametrize(
    "method",
    [
        "/grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo",
        "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
    ],
)
@patch.dict(os.environ, {"DUO_WORKFLOW_GRPC_REFLECTION_ENABLED": "true"})
@pytest.mark.asyncio
async def test_intercept_service_reflection_skips_auth_when_enabled(
    method, mock_continuation, handler_call_details
):
    interceptor = AuthenticationInterceptor()
    handler_call_details.method = method
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    mock_continuation.assert_awaited_once_with(handler_call_details)


@pytest.mark.parametrize(
    "method",
    [
        "/grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo",
        "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
    ],
)
@patch(
    "duo_workflow_service.interceptors.authentication_interceptor.authenticate",
    return_value=(None, MagicMock(error_message="No authorization header presented")),
)
@pytest.mark.asyncio
async def test_intercept_service_reflection_requires_auth_by_default(
    mock_authenticate, method, mock_continuation, handler_call_details
):
    interceptor = AuthenticationInterceptor()
    handler_call_details.method = method
    handler_call_details.invocation_metadata = ()
    await interceptor.intercept_service(mock_continuation, handler_call_details)
    mock_continuation.assert_not_awaited()


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
