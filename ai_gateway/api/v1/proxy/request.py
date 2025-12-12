import functools
import typing

from fastapi import BackgroundTasks, HTTPException, Request, status
from gitlab_cloud_connector import (
    FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    UNIT_PRIMITIVE_AND_DESCRIPTION_MAPPING,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
)

from ai_gateway.abuse_detection import AbuseDetector
from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from lib.billing_events.client import BillingEventsClient
from lib.internal_events.context import EventContext, current_event_context

# It's implemented here, because eventually we want to restrict this endpoint to
# ai_gateway_model_provider_proxy unit primitive only, so we won't rely on
# FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS const anymore.
#
# https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1420
#
# Currently, this endpoint is used by older self-managed instances, so we cannot just restrict
# the list of unit primitives due to the backward compatibility promise.
EXTENDED_FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS = {
    **FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    **{
        GitLabUnitPrimitive.AI_GATEWAY_MODEL_PROVIDER_PROXY: GitLabFeatureCategory.DUO_AGENT_PLATFORM
    },
}


def authorize_with_unit_primitive_header():
    """Authorize with x-gitlab-unit-primitive header.

    See
    https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/auth.md#use-x-gitlab-unit-primitive-header
    for more information.
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            background_tasks: BackgroundTasks,
            abuse_detector: AbuseDetector,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            await _validate_request(request, background_tasks, abuse_detector)
            return await func(
                request, background_tasks, abuse_detector, *args, **kwargs
            )

        return wrapper

    return decorator


async def _validate_request(
    request: Request, background_tasks: BackgroundTasks, abuse_detector: AbuseDetector
) -> None:
    if X_GITLAB_UNIT_PRIMITIVE not in request.headers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing {X_GITLAB_UNIT_PRIMITIVE} header",
        )

    unit_primitive = request.headers[X_GITLAB_UNIT_PRIMITIVE]

    if unit_primitive not in GitLabUnitPrimitive.__members__.values():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown unit primitive header {unit_primitive}",
        )

    unit_primitive = GitLabUnitPrimitive(unit_primitive)

    current_user: StarletteUser = request.user

    if not current_user.can(unit_primitive):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unauthorized to access {unit_primitive}",
        )

    if abuse_detector.should_detect():
        body_bytes = await request.body()
        body = body_bytes.decode("utf-8", errors="ignore")

        description = UNIT_PRIMITIVE_AND_DESCRIPTION_MAPPING.get(unit_primitive, "")
        background_tasks.add_task(abuse_detector.detect, request, body, description)


def track_billing_event(func):
    @functools.wraps(func)
    async def wrapper(
        request: Request,
        *args: typing.Any,
        billing_event_client: BillingEventsClient,
        **kwargs: typing.Any,
    ) -> typing.Any:
        response = await func(
            request, *args, billing_event_client=billing_event_client, **kwargs
        )

        metadata = None
        if hasattr(response, "body") and response.status_code == 200:
            try:
                # Check if the proxy client stored TokenUsage in request state
                if hasattr(request.state, "proxy_token_usage") and hasattr(
                    request.state, "proxy_model_name"
                ):
                    token_usage = request.state.proxy_token_usage
                    model_id = request.state.proxy_model_name

                    llm_operation = {
                        "model_id": model_id,
                        **token_usage.to_billing_metadata(),
                    }
                    metadata = {"llm_operations": [llm_operation]}
            except KeyError:
                # If we can't parse the response, continue without metadata
                pass

        # Track event only after `func` returns so we don't trigger a billable event if an exception occurred
        billing_event_client.track_billing_event(
            request.user,
            event_type="ai_gateway_proxy_use",
            category=__name__,
            unit_of_measure="request",
            quantity=1,
            metadata=metadata,
        )

        return response

    return wrapper


def verify_project_namespace_metadata():
    """Verify that project and namespace headers matches the claims from the token."""

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            internal_context: EventContext = current_event_context.get()
            user_claims = request.user.claims

            extra_claims = user_claims.extra or {}

            if str(internal_context.project_id) != str(
                extra_claims.get("gitlab_project_id", None)
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Gitlab project id mismatch",
                )

            if str(internal_context.namespace_id) != str(
                extra_claims.get("gitlab_namespace_id", None)
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Gitlab namespace id mismatch",
                )

            if str(internal_context.ultimate_parent_namespace_id) != str(
                extra_claims.get("gitlab_root_namespace_id", None)
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Gitlab root namespace id mismatch",
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
