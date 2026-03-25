import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)


@pytest.fixture(name="mock_user")
def mock_user_fixture():
    return CloudConnectorUser(True, claims=UserClaims(gitlab_realm="test-realm"))


@pytest.mark.asyncio
async def test_model_metadata_interceptor_sets_metadata(mock_user):
    """Test that the interceptor processes model metadata correctly."""
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        (
            "x-gitlab-agent-platform-model-metadata",
            json.dumps(
                {
                    "model": "claude-3-5-sonnet-20240620",
                    "version": "1.0",
                    "provider": "anthropic",
                }
            ),
        ),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock(return_value="mocked_response")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata_by_size"
        ) as mock_create,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
    ):
        mock_metadata_by_size = MagicMock()
        mock_create.return_value = mock_metadata_by_size

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_create.assert_called_once_with(
            {
                "model": "claude-3-5-sonnet-20240620",
                "version": "1.0",
                "provider": "anthropic",
            }
        )
        mock_context.set.assert_called_once_with(mock_metadata_by_size.default)
        mock_size_context.set.assert_called_once_with(mock_metadata_by_size)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
async def test_model_metadata_interceptor_null_json_skips_context_setting(mock_user):
    """JSON 'null' parses to None, which create_model_metadata_by_size rejects with ValueError.

    Both context vars must remain unset and the request must still continue.
    """
    current_user.set(mock_user)
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-agent-platform-model-metadata", "null"),
    ]

    continuation = AsyncMock(return_value="mocked_response")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_context.set.assert_not_called()
        mock_size_context.set.assert_not_called()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invocation_metadata,_test_case",
    [
        ([("other-header", "other-value")], "no_metadata_header"),
        (
            [
                ("x-gitlab-agent-platform-model-metadata", ""),
                ("other-header", "other-value"),
            ],
            "empty_metadata",
        ),
        (
            [
                ("x-gitlab-agent-platform-model-metadata", "invalid-json{"),
                ("other-header", "other-value"),
            ],
            "invalid_json",
        ),
    ],
)
async def test_model_metadata_interceptor_no_processing_scenarios(
    invocation_metadata, _test_case
):
    """Test that the interceptor handles cases where no model metadata processing occurs."""
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = invocation_metadata

    continuation = AsyncMock(return_value="mocked_response")

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata_by_size"
        ) as mock_create,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_with_size_context"
        ) as mock_size_context,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_create.assert_not_called()
        mock_context.set.assert_not_called()
        mock_size_context.set.assert_not_called()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"
