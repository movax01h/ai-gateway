from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from duo_workflow_service.interceptors.prompt_caching_interceptor import (
    PromptCachingInterceptor,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header_value",
    [
        "true",
        "false",
    ],
)
async def test_model_metadata_interceptor_processing_scenarios(header_value):
    interceptor = PromptCachingInterceptor()

    handler_call_details = MagicMock()
    handler_call_details.invocation_metadata = {
        "x-gitlab-model-prompt-cache-enabled": header_value
    }

    continuation = AsyncMock()
    continuation.return_value = "mocked_response"

    with (
        patch(
            "duo_workflow_service.interceptors.prompt_caching_interceptor.set_prompt_caching_enabled_to_current_request"
        ) as mock_context,
    ):
        result = await interceptor.intercept_service(continuation, handler_call_details)

        mock_context.assert_called_once_with(header_value)
        continuation.assert_called_once_with(handler_call_details)
        assert result == "mocked_response"
