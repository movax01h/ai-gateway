from typing import Optional

from ai_gateway.models.base import ModelAPICallError


def extract_retry_after_header(exc: ModelAPICallError) -> Optional[str]:
    retry_after = None
    if hasattr(exc, "errors") and exc.errors:
        original_error = exc.errors[0]
        if hasattr(original_error, "response"):
            retry_after = original_error.response.headers.get("retry-after")

    return retry_after
