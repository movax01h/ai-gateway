"""Custom exceptions for usage quota operations."""


class UsageQuotaError(Exception):
    """Base exception for usage quota related errors."""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self):
        return self.message


class UsageQuotaTimeoutError(UsageQuotaError):
    """Raised when the Usage Quota API request times out."""

    def __init__(self, original_error: Exception | None = None):
        super().__init__(
            "Request to Usage Quota API timed out",
            original_error=original_error,
        )


class UsageQuotaHTTPError(UsageQuotaError):
    """Raised when the Usage Quota API returns an unexpected HTTP error."""

    def __init__(self, status_code: int, original_error: Exception | None = None):
        self.status_code = status_code
        super().__init__(
            f"Usage Quota API returned HTTP error: {status_code}",
            original_error=original_error,
        )


class UsageQuotaConnectionError(UsageQuotaError):
    """Raised when there's a connection error to the Usage Quota API."""

    def __init__(self, original_error: Exception | None = None):
        super().__init__(
            "Failed to connect to Usage Quota API",
            original_error=original_error,
        )
