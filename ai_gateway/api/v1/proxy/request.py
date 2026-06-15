import functools
import typing

from dependency_injector.wiring import Provide, inject
from fastapi import HTTPException, Request, status
from gitlab_cloud_connector import (
    FEATURE_CATEGORIES_FOR_PROXY_ENDPOINTS,
    GitLabFeatureCategory,
    GitLabUnitPrimitive,
    UserClaims,
)

from ai_gateway.api.feature_category import X_GITLAB_UNIT_PRIMITIVE
from ai_gateway.config import ConfigProxyEndpoints
from ai_gateway.container import ContainerApplication
from lib.context import StarletteUser
from lib.internal_events.context import current_event_context

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


@inject
async def _check_proxy_endpoints_enabled(
    request: Request,
    proxy_cfg: dict = Provide[ContainerApplication.config.proxy_endpoints],
) -> None:
    """Raise HTTP 402 if the request realm is not on the proxy endpoint allowlist.

    Reads ``AIGW_PROXY_ENDPOINTS__SELF_MANAGED_ENABLED`` and
    ``AIGW_PROXY_ENDPOINTS__SAAS_ENABLED`` from the DI config. Each accepts:

    - Absent or empty string → all instances/namespaces are allowed (no restriction).
    - ``"id1,id2"`` → only those instance UIDs (self-managed) or root namespace IDs (SaaS) are allowed.

    SaaS realms are checked against ``saas_enabled``; all other realms (including
    ``self-managed``, unrecognised realms, and empty realm) are checked against
    ``self_managed_enabled``.
    """
    user_claims = request.user.claims or UserClaims()
    realm = user_claims.gitlab_realm or ""

    cfg = (
        ConfigProxyEndpoints(**proxy_cfg) if isinstance(proxy_cfg, dict) else proxy_cfg
    )

    if realm == "saas":
        allowed_ids = cfg.saas_enabled_ids()
        if allowed_ids:
            extra_claims = user_claims.extra or {}
            namespace_id = str(extra_claims.get("gitlab_root_namespace_id") or "")
            if namespace_id not in allowed_ids:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Proxy endpoints are not enabled for this namespace",
                )
    else:
        allowed_ids = cfg.self_managed_enabled_ids()
        if allowed_ids:
            instance_id = str(user_claims.gitlab_instance_uid or "")
            if instance_id not in allowed_ids:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Proxy endpoints are not enabled for this instance",
                )


def check_proxy_endpoints_enabled():
    """Decorator enforcing the proxy endpoint allowlist via DI-injected config.

    See ``_check_proxy_endpoints_enabled`` for the enforcement logic.
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            await _check_proxy_endpoints_enabled(request)
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


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
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            await _validate_request(request)
            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


async def _validate_request(request: Request) -> None:
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


def verify_project_namespace_metadata():
    """Verify that project and namespace headers matches the claims from the token.

    For SaaS instances, verifies project, namespace, and root namespace IDs. For self-managed instances, verifies only
    the instance UID.
    """

    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(
            request: Request,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            internal_context = current_event_context.get()
            user_claims = request.user.claims or UserClaims()

            extra_claims = user_claims.extra or {}

            # For self-managed instances, verify instance UID only
            if user_claims.gitlab_realm == "self-managed":
                if str(internal_context.instance_id) != str(
                    user_claims.gitlab_instance_uid
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Gitlab instance uid mismatch",
                    )
            # For SaaS instances, verify project, namespace, and root namespace IDs
            else:
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
