import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duo_workflow_service.interceptors.model_metadata_interceptor import (
    ModelMetadataInterceptor,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata_value,expected_data,test_case",
    [
        (
            json.dumps(
                {
                    "model": "claude-3-5-sonnet-20240620",
                    "version": "1.0",
                    "provider": "anthropic",
                }
            ),
            {
                "model": "claude-3-5-sonnet-20240620",
                "version": "1.0",
                "provider": "anthropic",
            },
            "sets_metadata",
        ),
        ("null", None, "null_json"),
    ],
)
async def test_model_metadata_interceptor_processing_scenarios(
    metadata_value, expected_data, test_case
):
    """Test that the interceptor processes model metadata correctly for valid scenarios."""
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = [
        ("x-gitlab-self-hosted-models-metadata", metadata_value),
        ("other-header", "other-value"),
    ]

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata"
        ) as mock_create,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
    ):

        mock_model_metadata = MagicMock()
        mock_create.return_value = mock_model_metadata

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_create.assert_called_once_with(expected_data)
        mock_context.set.assert_called_once_with(mock_model_metadata)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invocation_metadata,test_case",
    [
        ([("other-header", "other-value")], "no_metadata_header"),
        (
            [
                ("x-gitlab-self-hosted-models-metadata", ""),
                ("other-header", "other-value"),
            ],
            "empty_metadata",
        ),
        (
            [
                ("x-gitlab-self-hosted-models-metadata", "invalid-json{"),
                ("other-header", "other-value"),
            ],
            "invalid_json",
        ),
    ],
)
async def test_model_metadata_interceptor_no_processing_scenarios(
    invocation_metadata, test_case
):
    """Test that the interceptor handles cases where no model metadata processing occurs."""
    interceptor = ModelMetadataInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = invocation_metadata

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.create_model_metadata"
        ) as mock_create,
        patch(
            "duo_workflow_service.interceptors.model_metadata_interceptor.current_model_metadata_context"
        ) as mock_context,
    ):

        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_create.assert_not_called()
        mock_context.set.assert_not_called()
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"
