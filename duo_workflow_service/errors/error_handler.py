import asyncio
from enum import Enum

import structlog

logger = structlog.stdlib.get_logger("error_handler")


class ModelErrorType(Enum):
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    REQUEST_TOO_LARGE = "request_too_large"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED_ERROR = "overloaded_error"
    UNKNOWN = "unknown"


ERROR_TYPES = {
    400: ModelErrorType.INVALID_REQUEST_ERROR,
    401: ModelErrorType.AUTHENTICATION_ERROR,
    403: ModelErrorType.PERMISSION_ERROR,
    404: ModelErrorType.NOT_FOUND_ERROR,
    413: ModelErrorType.REQUEST_TOO_LARGE,
    429: ModelErrorType.RATE_LIMIT_ERROR,
    500: ModelErrorType.API_ERROR,
    529: ModelErrorType.OVERLOADED_ERROR,
}


RETRYABLE_ERRORS = {
    ModelErrorType.RATE_LIMIT_ERROR,
    ModelErrorType.API_ERROR,
    ModelErrorType.OVERLOADED_ERROR,
}


class ModelError(Exception):
    def __init__(
        self,
        error_type: ModelErrorType,
        status_code: int,
        message: str,
    ):
        self.error_type = error_type
        self.status_code = status_code
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.error_type.value}: {self.message} (Status: {self.status_code})"


class ModelErrorHandler:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._retry_count: int = 0

    def get_error_type(self, status_code: int) -> ModelErrorType:
        return ERROR_TYPES.get(status_code, ModelErrorType.UNKNOWN)

    def _get_retry_after(self) -> float:
        return self.base_delay * (2**self._retry_count)

    async def handle_error(self, error: ModelError) -> None:
        if error.error_type not in RETRYABLE_ERRORS:
            raise error

        if self._retry_count >= self.max_retries:
            raise error

        retry_after = self._get_retry_after()

        logger.warn(
            "Anthropic API error occurred. Retrying.",
            error=str(error),
            attempt=self._retry_count + 1,
            max_retries=self.max_retries,
            retry_after=retry_after,
        )

        self._retry_count += 1
        await asyncio.sleep(retry_after)
