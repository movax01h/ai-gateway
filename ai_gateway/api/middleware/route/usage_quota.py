import functools
from typing import Any, Callable, Optional

import structlog
from dependency_injector.wiring import Provide, inject
from starlette.requests import Request
from starlette.responses import JSONResponse

from ai_gateway.container import ContainerApplication
from lib.context import StarletteUser
from lib.events import FeatureQualifiedNameStatic, GLReportingEventContext
from lib.usage_quota import InsufficientCredits, ModelMetadata, UsageQuotaEvent
from lib.usage_quota.client import should_skip_usage_quota_for_user
from lib.usage_quota.service import UsageQuotaService

log = structlog.stdlib.get_logger("usage_quota")


def has_sufficient_usage_quota(
    feature_qualified_name: FeatureQualifiedNameStatic,
    event: UsageQuotaEvent | Callable[[Any], Any],
    model_name: str | None = None,
):
    """Decorator to enforce usage quota checks on API routes.

    This decorator should be applied to FastAPI route handlers that emit
    billable events. It checks the usage quota before executing the route
    and returns a 402 response if the user has insufficient credits.

    Args:
        feature_qualified_name: A static feature name enum value for this endpoint
            (e.g., FeatureQualifiedNameStatic.CODE_SUGGESTIONS). Required parameter.
        event: Either a UsageQuotaEvent enum value or a callable that resolves
            the event type from the request payload. The callable can be
            sync or async and should return a UsageQuotaEvent.
        model_name: Optional model name to include in usage quota metadata.

    Returns:
        A decorator function that wraps the route handler
    """

    def decorator(func: Callable) -> Callable:
        event_type_resolver = event if callable(event) else None
        static_event: Optional[UsageQuotaEvent] = event if not callable(event) else None
        return _process_route(
            func, feature_qualified_name, static_event, event_type_resolver, model_name
        )

    return decorator


async def _resolve_event_type_from_request(
    event_type_resolver: Optional[Callable[[Any], Any]],
    kwargs: dict[str, Any],
) -> Optional[UsageQuotaEvent]:
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


@inject
async def _usage_quota_wrapper(
    func: Callable,
    request: Request,
    *args: Any,
    feature_qualified_name: FeatureQualifiedNameStatic,
    event: Optional[UsageQuotaEvent],
    event_type_resolver: Optional[Callable[[Any], Any]],
    model_name: Optional[str],
    usage_quota_service: UsageQuotaService = Provide[
        ContainerApplication.usage_quota.service
    ],
    **kwargs: Any,
) -> Any:
    current_user: StarletteUser = request.user

    if should_skip_usage_quota_for_user(current_user):
        return await func(request, *args, **kwargs)

    resolved_event = await _resolve_event_type_from_request(event_type_resolver, kwargs)
    event_to_use = resolved_event if resolved_event else event

    if not event_to_use:
        log.info("No event available")
        return await func(request, *args, **kwargs)

    gl_reporting_context = GLReportingEventContext.from_static_name(
        feature_qualified_name, is_ai_catalog_item=False
    )

    try:
        await usage_quota_service.execute(
            gl_reporting_context,
            event_to_use,
            model_metadata=ModelMetadata(name=model_name) if model_name else None,
        )
    except InsufficientCredits:
        return _insufficient_credits_response()

    return await func(request, *args, **kwargs)


def _process_route(
    func: Callable,
    feature_qualified_name: FeatureQualifiedNameStatic,
    event: Optional[UsageQuotaEvent],
    event_type_resolver: Optional[Callable[[Any], Any]],
    model_name: Optional[str],
) -> Callable:
    @functools.wraps(func)
    async def wrapper(
        request: Request,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return await _usage_quota_wrapper(
            func,
            request,
            *args,
            feature_qualified_name=feature_qualified_name,
            event=event,
            event_type_resolver=event_type_resolver,
            model_name=model_name,
            **kwargs,
        )

    return wrapper
