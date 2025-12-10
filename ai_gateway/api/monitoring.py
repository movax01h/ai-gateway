import functools
from typing import Annotated, Any, Awaitable, Callable

from dependency_injector.providers import Configuration, Factory
from fastapi import APIRouter, Depends, Request
from fastapi_health import health
from gitlab_cloud_connector import cloud_connector_ready

from ai_gateway.async_dependency_resolver import (
    get_code_suggestions_completions_litellm_factory_provider,
    get_config,
    get_prompt_registry,
)
from ai_gateway.code_suggestions.completions import CodeCompletions
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.litellm import KindLiteLlmModel
from ai_gateway.prompts.base import BasePromptRegistry

__all__ = [
    "router",
]

router = APIRouter(
    prefix="/monitoring",
    tags=["monitoring"],
)

# Avoid calling out to the models multiple times from this public, unauthenticated endpoint.
# this is not threadsafe, but that should be fine, we aren't issuing multiple of
# these calls in parallel. When the instance is marked as ready, we won't be modifying
# the list anymore.
validated: set[KindModelProvider] = set()


def single_validation(
    key: KindModelProvider,
):
    def _decorator(
        func: Callable[[Any], Awaitable[bool]],
    ) -> Callable[[Any, Any], Awaitable[bool]]:

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs) -> bool:
            if key in validated:
                return True

            result = await func(*args, **kwargs)
            validated.add(key)

            return result

        return _wrapper

    return _decorator


async def validate_default_models_available(
    prompt_registry: Annotated[BasePromptRegistry, Depends(get_prompt_registry)],
) -> bool:
    return await prompt_registry.validate_default_models()


@single_validation(KindModelProvider.FIREWORKS)
async def validate_fireworks_available(
    completions_litellm_factory: Annotated[
        Factory[CodeCompletions],
        Depends(get_code_suggestions_completions_litellm_factory_provider),
    ],
) -> bool:
    code_completions = completions_litellm_factory(
        model__name=KindLiteLlmModel.QWEN_2_5,
        model__provider=KindModelProvider.FIREWORKS,
    )
    await code_completions.execute(
        prefix="def hello_world():",
        suffix="",
        file_name="monitoring.py",
        editor_lang="python",
    )
    return True


async def validate_cloud_connector_ready(
    config: Annotated[Configuration, Depends(get_config)],
    request: Request,
) -> bool:
    """Always pass for Self-Hosted-Models.

    This is temporary. With the current CC <-> AI GW interface, we can't easily skip CDot sync just for SHM only. At the
    same time, we can't require SHM to always connect to CustomersDot - as CustomersDot connection/sync is not needed
    for AI GW to work in SHM context. So we shouldn't fail the probe for customer setups that can't or don't want to
    reach CustomersDot. As soon as
    https://gitlab.com/gitlab-org/gitlab/-/issues/517088
    is complete, we should stop passing CustomersDot
    as a provider for SHM setups. With that, we can drop the check for SHM in this file.
    """
    if config.custom_models.enabled():
        return True

    provider = request.app.state.cloud_connector_auth_provider
    return cloud_connector_ready(provider)


router.add_api_route("/healthz", health([]))
router.add_api_route(
    "/ready",
    health(
        [
            validate_fireworks_available,
            validate_default_models_available,
            validate_cloud_connector_ready,
        ]
    ),
)
