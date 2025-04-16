import pytest

from duo_workflow_service.errors.error_handler import (
    ERROR_TYPES,
    RETRYABLE_ERRORS,
    ModelError,
    ModelErrorHandler,
)


@pytest.mark.asyncio
async def test_handle_non_retryable_error():
    handler = ModelErrorHandler()

    non_retryable_codes = {
        code: error_type
        for code, error_type in ERROR_TYPES.items()
        if error_type not in RETRYABLE_ERRORS
    }

    for status_code, error_type in non_retryable_codes.items():
        error = ModelError(
            error_type=error_type, status_code=status_code, message="Error message"
        )

        with pytest.raises(ModelError) as exc_info:
            await handler.handle_error(error)

        assert exc_info.value.error_type == error_type
        assert handler._retry_count == 0


@pytest.mark.asyncio
async def test_handle_retryable_error():
    handler = ModelErrorHandler(max_retries=2, base_delay=0.1)

    retryable_error_types = {
        code: error_type
        for code, error_type in ERROR_TYPES.items()
        if error_type in RETRYABLE_ERRORS
    }

    for status_code, error_type in retryable_error_types.items():
        handler = ModelErrorHandler(max_retries=2, base_delay=0.1)
        error = ModelError(
            error_type=error_type,
            status_code=status_code,
            message="Error message",
        )

        await handler.handle_error(error)
        assert handler._retry_count == 1

        await handler.handle_error(error)
        assert handler._retry_count == 2

        with pytest.raises(ModelError) as exc_info:
            await handler.handle_error(error)

        assert exc_info.value.error_type == error_type
