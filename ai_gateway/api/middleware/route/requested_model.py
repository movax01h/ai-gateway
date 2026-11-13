import functools
from typing import Any, Callable

from starlette_context import context as starlette_context

_REQUESTED_MODEL_PROVIDER_CONTEXT_KEY = "meta.requested_model_provider"
_REQUESTED_MODEL_NAME_CONTEXT_KEY = "meta.requested_model_name"


def record_requested_model() -> Callable[[Callable], Callable]:
    """Record the model the client requested in the request context, so `AccessLogMiddleware` logs it.

    The request payload is passed through the `payload` keyword argument and must be a
    `SuggestionsRequest` instance. Its `model_provider` and `model_name` fields are
    recorded in the request context. Other payload types are silently ignored.

    Place it above `has_sufficient_usage_quota` so requests denied by the quota check are covered too. It must run
    before the handler replaces the payload's model with the resolved one. Requests that fail FastAPI's request body
    validation (422) never reach it, so they aren't covered.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from ai_gateway.api.v2.code.typing import SuggestionsRequest

            payload = kwargs.get("payload")
            if isinstance(payload, SuggestionsRequest):
                starlette_context[_REQUESTED_MODEL_PROVIDER_CONTEXT_KEY] = (
                    payload.model_provider
                )
                starlette_context[_REQUESTED_MODEL_NAME_CONTEXT_KEY] = (
                    payload.model_name
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator
