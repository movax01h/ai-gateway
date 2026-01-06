import functools
from typing import Any, Callable, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

from lib.events import GLReportingEventContext
from lib.usage_quota import EventType, InsufficientCredits


def has_sufficient_usage_quota(
    feature_qualified_name: str,
    event: EventType | Callable[[Any], Any],
):
    """Decorator to enforce usage quota checks on API routes.

    This decorator should be applied to FastAPI route handlers that emit
    billable events. It checks the usage quota before executing the route
    and returns a 402 response if the user has insufficient credits.

    Args:
        feature_qualified_name: The feature name for this endpoint
            (e.g., "code_suggestions"). Required parameter.
        event: Either an EventType enum value or a callable that resolves
            the event type from the request payload. The callable can be
            sync or async and should return an EventType.

    Returns:
        A decorator function that wraps the route handler
    """

    def decorator(func: Callable) -> Callable:
        event_type_resolver = event if callable(event) else None
        static_event: Optional[EventType] = event if not callable(event) else None
        return _process_route(
            func, feature_qualified_name, static_event, event_type_resolver
        )

    return decorator


async def _resolve_event_type_from_request(
    event_type_resolver: Optional[Callable[[Any], Any]],
    kwargs: dict[str, Any],
) -> Optional[EventType]:
    if event_type_resolver:
        try:
            # Extract the payload from kwargs (Pydantic model without Query/Path/Depends)
            payload = next(
                (v for v in kwargs.values() if hasattr(v, "model_dump")),
                None,
            )
            result = event_type_resolver(payload)
            # Handle both sync and async callables
            if hasattr(result, "__await__"):
                return await result
            return result
        except Exception:
            pass

    return None


def _insufficient_credits_response() -> JSONResponse:
    return JSONResponse(
        status_code=402,
        content={
            "error": "insufficient_credits",
            "error_code": "USAGE_QUOTA_EXCEEDED",
            "message": ("Consumer does not have sufficient credits for this request"),
        },
    )


def _process_route(
    func: Callable,
    feature_qualified_name: str,
    event: Optional[EventType],
    event_type_resolver: Optional[Callable[[Any], Any]],
) -> Callable:
    @functools.wraps(func)
    async def wrapper(
        request: Request,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        service = request.app.state.usage_quota_service
        resolved_event = await _resolve_event_type_from_request(
            event_type_resolver, kwargs
        )
        event_to_use = resolved_event if resolved_event else event

        gl_reporting_context = GLReportingEventContext.from_workflow_definition(
            feature_qualified_name
        )

        try:
            await service.execute(gl_reporting_context, event_to_use)
        except InsufficientCredits:
            return _insufficient_credits_response()

        return await func(request, *args, **kwargs)

    return wrapper
